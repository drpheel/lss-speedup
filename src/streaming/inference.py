"""Background inference runtime for live camera + BEV streams."""

import time
from typing import List, Optional

import cv2
import numpy as np
import torch

from src.data import compile_data
from src.models import LiftSplatShoot

from .config import (
    CACHE_BATCHES,
    DATA_AUG_CONF,
    DATAROOT,
    ENABLE_LIDAR_BEV_EVAL,
    EVAL_THRESHOLD,
    FRAME_INTERVAL,
    GRID_CONF,
    JPEG_QUALITY,
    LOG_EVERY_SEC,
    LOOP_DATASET,
    NUSCENES_VERSION,
    PREFETCH_FACTOR,
    SHOW_OVERLAY,
    TENSORRT_BEV_ENGINE_PATH,
    TENSORRT_CAM_ENGINE_PATH,
    USE_FP16,
    USE_TENSORRT,
    WEIGHTS_PATH,
)
from .eval_metrics import RunningBevEval
from .state import SharedState
from .tensorrt_utils import maybe_build_single_input_tensorrt
from .visualization import draw_ego_triangle, draw_overlay, maybe_resize, tile_cameras


class BatchProvider:
    def __init__(self, val_loader, cached_batches: List[tuple], loop_dataset: bool):
        self.val_loader = val_loader
        self.cached_batches = cached_batches
        self.loop_dataset = loop_dataset
        self.cached_idx = 0
        self.val_iter = iter(val_loader)

    def next_batch(self):
        if self.cached_batches:
            batch = self.cached_batches[self.cached_idx]
            self.cached_idx = (self.cached_idx + 1) % len(self.cached_batches)
            return batch, 0.0, False

        t_fetch0 = time.perf_counter()
        try:
            batch = next(self.val_iter)
        except StopIteration:
            if not self.loop_dataset:
                return None, 0.0, True
            self.val_iter = iter(self.val_loader)
            batch = next(self.val_iter)

        fetch_ms = (time.perf_counter() - t_fetch0) * 1000.0
        return batch, fetch_ms, False


def _init_model_and_loader():
    print("Initializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = LiftSplatShoot(GRID_CONF, DATA_AUG_CONF, outC=1)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.to(device)
    model.eval()
    if hasattr(model, "profile_voxel_pooling"):
        model.profile_voxel_pooling = False

    print(f"Loading nuScenes {NUSCENES_VERSION} split from {DATAROOT}...")
    _, val_loader = compile_data(
        NUSCENES_VERSION,
        DATAROOT,
        DATA_AUG_CONF,
        GRID_CONF,
        bsz=1,
        nworkers=2,
        parser_name="segmentationdata",
    )
    val_dataset = val_loader.dataset
    print(f"Validation samples available: {len(val_dataset)}")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )
    return model, device, val_loader, val_dataset


def _cache_batches_if_requested(val_loader, val_dataset) -> List[tuple]:
    cached_batches: List[tuple] = []
    if CACHE_BATCHES <= 0:
        return cached_batches

    print(f"Caching {CACHE_BATCHES} validation batches in RAM...")
    for i, batch in enumerate(val_loader):
        cached_batches.append(batch)
        if i + 1 >= CACHE_BATCHES:
            break

    print(f"Cached {len(cached_batches)} batches.")
    if len(cached_batches) < len(val_dataset):
        print(
            "Note: replaying cached subset for speed. "
            "Set CACHE_BATCHES=0 for full-split sequential evaluation."
        )
    return cached_batches


def _move_batch_to_device(batch, device):
    imgs, rots, trans, intrinsics, post_rots, post_trans, binimgs = batch
    t_h2d0 = time.perf_counter()
    imgs = imgs.to(device, non_blocking=True)
    rots = rots.to(device, non_blocking=True)
    trans = trans.to(device, non_blocking=True)
    intrinsics = intrinsics.to(device, non_blocking=True)
    post_rots = post_rots.to(device, non_blocking=True)
    post_trans = post_trans.to(device, non_blocking=True)
    t_h2d_ms = (time.perf_counter() - t_h2d0) * 1000.0
    return (imgs, rots, trans, intrinsics, post_rots, post_trans, binimgs, t_h2d_ms)


def _backend_label(using_trt_cam: bool, using_trt_bev: bool) -> str:
    if using_trt_cam and using_trt_bev:
        return "TRT(cam+bev)"
    if using_trt_cam:
        return "TRT(cam)"
    if using_trt_bev:
        return "TRT(bev)"
    return "Torch"


def _update_eval_stats(
    shared_state: SharedState,
    eval_tracker: Optional[RunningBevEval],
    backend_label: str,
) -> None:
    if eval_tracker is None:
        return

    eval_summary = eval_tracker.summary()
    shared_state.set_eval_stats(
        {
            "ready": True,
            "backend": backend_label,
            "timestamp_unix": time.time(),
            **eval_summary,
        }
    )

    print(
        f"[EVAL] frames={eval_summary['frames']} "
        f"IoU={eval_summary['iou']:.3f} P={eval_summary['precision']:.3f} "
        f"R={eval_summary['recall']:.3f} F1={eval_summary['f1']:.3f} "
        f"pos_diff={eval_summary['pos_rate_diff']:+.4f} "
        f"(last_iou={eval_summary['last_iou']:.3f})"
    )
    bins = eval_summary["distance_bins"]
    bin_iou_text = " ".join(f"{label}={vals['iou']:.3f}" for label, vals in bins.items())
    print(
        "[EVAL-BINS] "
        f"{bin_iou_text} "
        f"| near_w_iou={eval_summary['nearfield_weighting']['weighted_iou']:.3f}"
    )


def run_inference_loop(shared_state: SharedState) -> None:
    model, device, val_loader, val_dataset = _init_model_and_loader()
    cached_batches = _cache_batches_if_requested(val_loader, val_dataset)
    provider = BatchProvider(val_loader, cached_batches, LOOP_DATASET)

    print("Inference Loop Started.")

    fps_ema = 0.0
    enc_ms_ema = 0.0
    fetch_ms_ema = 0.0
    copy_ms_ema = 0.0
    last_log_t = time.perf_counter()

    camencode_runner = model.camencode
    bevencode_runner = model.bevencode
    using_tensorrt_camencode = False
    using_tensorrt_bevencode = False
    trt_cam_attempted = False
    trt_bev_attempted = False

    eval_tracker = RunningBevEval(threshold=EVAL_THRESHOLD) if ENABLE_LIDAR_BEV_EVAL else None

    while True:
        batch, fetch_ms, done = provider.next_batch()
        if done:
            print("Reached end of validation split (single-pass mode).")
            break

        fetch_ms_ema = fetch_ms if fetch_ms_ema == 0.0 else (0.9 * fetch_ms_ema + 0.1 * fetch_ms)

        loop_t0 = time.perf_counter()
        imgs_cpu = batch[0][0]
        should_log = (time.perf_counter() - last_log_t) >= LOG_EVERY_SEC

        imgs, rots, trans, intrinsics, post_rots, post_trans, binimgs, t_h2d_ms = _move_batch_to_device(
            batch, device
        )

        if USE_TENSORRT and not trt_cam_attempted:
            trt_cam_attempted = True
            sample_cam_input = imgs.view(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4])
            camencode_runner, using_tensorrt_camencode = maybe_build_single_input_tensorrt(
                model.camencode,
                sample_cam_input,
                device,
                TENSORRT_CAM_ENGINE_PATH,
                "camencode",
            )

        t_infer0 = time.perf_counter()
        stage_t0 = stage_t1 = stage_t2 = stage_t3 = stage_t4 = None

        with torch.inference_mode():
            with torch.cuda.amp.autocast(
                enabled=(
                    USE_FP16
                    and device.type == "cuda"
                    and not using_tensorrt_camencode
                    and not using_tensorrt_bevencode
                )
            ):
                bsz, ncams, ch, im_h, im_w = imgs.shape

                if should_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    stage_t0 = time.perf_counter()

                geom = model.get_geometry(rots, trans, intrinsics, post_rots, post_trans)

                if should_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    stage_t1 = time.perf_counter()

                cam_in = imgs.view(bsz * ncams, ch, im_h, im_w)
                cam_feats = (
                    camencode_runner(cam_in) if using_tensorrt_camencode else model.camencode(cam_in)
                )

                if should_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    stage_t2 = time.perf_counter()

                cam_feats = cam_feats.view(
                    bsz,
                    ncams,
                    model.camC,
                    model.D,
                    im_h // model.downsample,
                    im_w // model.downsample,
                )
                cam_feats = cam_feats.permute(0, 1, 3, 4, 5, 2)

                vox = model.voxel_pooling(geom, cam_feats)

                if should_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    stage_t3 = time.perf_counter()

                if USE_TENSORRT and not trt_bev_attempted:
                    trt_bev_attempted = True
                    bevencode_runner, using_tensorrt_bevencode = maybe_build_single_input_tensorrt(
                        model.bevencode,
                        vox,
                        device,
                        TENSORRT_BEV_ENGINE_PATH,
                        "bevencode",
                    )

                bev_logits = bevencode_runner(vox) if using_tensorrt_bevencode else model.bevencode(vox)

                if should_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    stage_t4 = time.perf_counter()

                bev_prob_t = torch.sigmoid(bev_logits).squeeze()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_infer_ms = (time.perf_counter() - t_infer0) * 1000.0

        if eval_tracker is not None:
            bev_target = binimgs.to(device, non_blocking=True)
            eval_tracker.update(bev_logits, bev_target)

        t_copy0 = time.perf_counter()
        bev_prob = bev_prob_t.detach().cpu().numpy()
        t_copy_ms = (time.perf_counter() - t_copy0) * 1000.0
        copy_ms_ema = t_copy_ms if copy_ms_ema == 0.0 else (0.9 * copy_ms_ema + 0.1 * t_copy_ms)

        t_vis0 = time.perf_counter()
        cam_tiled = tile_cameras(imgs_cpu)
        bev_img = (bev_prob * 255).astype(np.uint8)
        bev_color = cv2.applyColorMap(bev_img, cv2.COLORMAP_JET)
        cam_tiled = maybe_resize(cam_tiled)
        bev_color = maybe_resize(bev_color)
        draw_ego_triangle(bev_color, GRID_CONF)
        t_vis_ms = (time.perf_counter() - t_vis0) * 1000.0

        loop_ms_so_far = (time.perf_counter() - loop_t0) * 1000.0
        inst_fps = 1000.0 / max(loop_ms_so_far + fetch_ms, 1e-6)
        fps_ema = inst_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst_fps)

        backend_label = _backend_label(using_tensorrt_camencode, using_tensorrt_bevencode)

        if should_log:
            geom_ms = cam_ms = voxel_ms = bev_ms = float("nan")
            if stage_t0 is not None and stage_t4 is not None:
                geom_ms = (stage_t1 - stage_t0) * 1000.0
                cam_ms = (stage_t2 - stage_t1) * 1000.0
                voxel_ms = (stage_t3 - stage_t2) * 1000.0
                bev_ms = (stage_t4 - stage_t3) * 1000.0

            print(
                f"[{backend_label}] fps={fps_ema:.1f} "
                f"infer_total={t_infer_ms:.1f}ms "
                f"geom={geom_ms:.1f}ms cam={cam_ms:.1f}ms "
                f"voxel={voxel_ms:.1f}ms bev={bev_ms:.1f}ms "
                f"copy={copy_ms_ema:.1f}ms"
            )

            _update_eval_stats(shared_state, eval_tracker, backend_label)
            last_log_t = time.perf_counter()

        if SHOW_OVERLAY:
            overlay_lines = [
                f"FPS(avg): {fps_ema:.1f} | FPS(inst): {inst_fps:.1f} | Backend: {backend_label}",
                f"Fetch(avg): {fetch_ms_ema:.1f} ms | H2D: {t_h2d_ms:.1f} ms",
                f"Infer: {t_infer_ms:.1f} ms | Copy(avg): {copy_ms_ema:.1f} ms | Vis: {t_vis_ms:.1f} ms | JPEG(avg): {enc_ms_ema:.1f} ms",
            ]
            draw_overlay(cam_tiled, overlay_lines)

        t_enc0 = time.perf_counter()
        cam_ok, cam_jpeg = cv2.imencode(
            ".jpg",
            cam_tiled,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        bev_ok, bev_jpeg = cv2.imencode(
            ".jpg",
            bev_color,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        t_enc_ms = (time.perf_counter() - t_enc0) * 1000.0
        enc_ms_ema = t_enc_ms if enc_ms_ema == 0.0 else (0.9 * enc_ms_ema + 0.1 * t_enc_ms)

        if not (cam_ok and bev_ok):
            continue

        shared_state.set_frames(cam_jpeg.tobytes(), bev_jpeg.tobytes())

        loop_dt = time.perf_counter() - loop_t0
        if loop_dt < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - loop_dt)
