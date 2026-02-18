#!/usr/bin/env python3
"""Reproducible benchmark harness for LSS backend comparison.

Compares two modes on identical cached validation batches:
- torch: PyTorch only
- trt:   TensorRT for camencode + bevencode (geometry/voxel remain PyTorch)

Writes a JSON report to make claims easy to verify and share.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.data import compile_data
from src.models import LiftSplatShoot
from src.streaming.config import (
    DATA_AUG_CONF,
    DATAROOT,
    GRID_CONF,
    NUSCENES_VERSION,
    TENSORRT_BEV_ENGINE_PATH,
    TENSORRT_CAM_ENGINE_PATH,
    USE_FP16,
    WEIGHTS_PATH,
)
from src.streaming.tensorrt_utils import HAS_TORCH2TRT, maybe_build_single_input_tensorrt


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    vals = sorted(values)
    k = (len(vals) - 1) * pct
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_model(device: torch.device) -> LiftSplatShoot:
    model = LiftSplatShoot(GRID_CONF, DATA_AUG_CONF, outC=1)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.to(device)
    model.eval()
    if hasattr(model, "profile_voxel_pooling"):
        model.profile_voxel_pooling = False
    return model


def build_cached_batches(device: torch.device, num_batches: int) -> List[tuple]:
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
    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        prefetch_factor=2,
    )

    cached = []
    for i, batch in enumerate(loader):
        cached.append(batch)
        if i + 1 >= num_batches:
            break
    if not cached:
        raise RuntimeError("No validation batches were cached; check dataset path and split.")
    return cached


def move_batch_to_device(batch: tuple, device: torch.device) -> tuple:
    imgs, rots, trans, intrinsics, post_rots, post_trans, binimgs = batch
    return (
        imgs.to(device, non_blocking=True),
        rots.to(device, non_blocking=True),
        trans.to(device, non_blocking=True),
        intrinsics.to(device, non_blocking=True),
        post_rots.to(device, non_blocking=True),
        post_trans.to(device, non_blocking=True),
        binimgs.to(device, non_blocking=True),
    )


def run_one_pass(
    model: LiftSplatShoot,
    device: torch.device,
    cached_batches: List[tuple],
    warmup_batches: int,
    measure_batches: int,
    mode: str,
) -> Dict[str, object]:
    assert mode in {"torch", "trt"}

    camencode_runner = model.camencode
    bevencode_runner = model.bevencode
    using_trt_cam = False
    using_trt_bev = False

    # Build/load TRT engines once using first batch tensor shapes.
    first_batch = move_batch_to_device(cached_batches[0], device)
    imgs, rots, trans, intrinsics, post_rots, post_trans, _ = first_batch

    if mode == "trt":
        if device.type != "cuda":
            raise RuntimeError("TRT mode requires CUDA device.")
        if not HAS_TORCH2TRT:
            raise RuntimeError("TRT mode requested but torch2trt is not installed.")

        sample_cam_input = imgs.view(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4])
        camencode_runner, using_trt_cam = maybe_build_single_input_tensorrt(
            model.camencode,
            sample_cam_input,
            device,
            TENSORRT_CAM_ENGINE_PATH,
            "camencode",
        )

    stage_times: Dict[str, List[float]] = {
        "geometry_ms": [],
        "camencode_ms": [],
        "voxel_pool_ms": [],
        "bevencode_ms": [],
        "total_ms": [],
    }

    total_iters = warmup_batches + measure_batches
    for i in range(total_iters):
        batch = cached_batches[i % len(cached_batches)]
        imgs, rots, trans, intrinsics, post_rots, post_trans, _ = move_batch_to_device(batch, device)

        sync_if_cuda(device)
        t0 = time.perf_counter()

        with torch.inference_mode():
            with torch.cuda.amp.autocast(
                enabled=(USE_FP16 and device.type == "cuda" and mode == "torch")
            ):
                bsz, ncams, ch, im_h, im_w = imgs.shape

                sync_if_cuda(device)
                s0 = time.perf_counter()
                geom = model.get_geometry(rots, trans, intrinsics, post_rots, post_trans)

                sync_if_cuda(device)
                s1 = time.perf_counter()
                cam_in = imgs.view(bsz * ncams, ch, im_h, im_w)
                cam_feats = camencode_runner(cam_in) if using_trt_cam else model.camencode(cam_in)

                sync_if_cuda(device)
                s2 = time.perf_counter()
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

                sync_if_cuda(device)
                s3 = time.perf_counter()

                if mode == "trt" and not using_trt_bev:
                    bevencode_runner, using_trt_bev = maybe_build_single_input_tensorrt(
                        model.bevencode,
                        vox,
                        device,
                        TENSORRT_BEV_ENGINE_PATH,
                        "bevencode",
                    )

                _ = bevencode_runner(vox) if using_trt_bev else model.bevencode(vox)

                sync_if_cuda(device)
                s4 = time.perf_counter()

        if i < warmup_batches:
            continue

        stage_times["geometry_ms"].append((s1 - s0) * 1000.0)
        stage_times["camencode_ms"].append((s2 - s1) * 1000.0)
        stage_times["voxel_pool_ms"].append((s3 - s2) * 1000.0)
        stage_times["bevencode_ms"].append((s4 - s3) * 1000.0)
        stage_times["total_ms"].append((s4 - t0) * 1000.0)

    summary = {}
    for k, vals in stage_times.items():
        summary[k] = {
            "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
            "p95": percentile(vals, 0.95),
        }

    mean_total_ms = summary["total_ms"]["mean"]
    summary["fps_from_mean_total"] = 1000.0 / max(mean_total_ms, 1e-6)
    summary["mode"] = mode
    summary["using_trt_camencode"] = using_trt_cam
    summary["using_trt_bevencode"] = using_trt_bev
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Torch vs TensorRT backends.")
    parser.add_argument("--num-batches", type=int, default=24, help="Batches to cache from val split.")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup iterations per mode.")
    parser.add_argument("--measure", type=int, default=32, help="Measured iterations per mode.")
    parser.add_argument(
        "--modes",
        type=str,
        default="torch,trt",
        help="Comma-separated modes: torch,trt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in {"torch", "trt"}:
            raise ValueError(f"Unsupported mode: {m}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark device: {device}")
    print(f"Caching {args.num_batches} validation batches from {DATAROOT}...")
    cached_batches = build_cached_batches(device, args.num_batches)

    results = {
        "timestamp_unix": time.time(),
        "device": str(device),
        "dataroot": DATAROOT,
        "weights_path": WEIGHTS_PATH,
        "num_cached_batches": len(cached_batches),
        "warmup_iters": args.warmup,
        "measure_iters": args.measure,
        "modes": {},
    }

    for mode in modes:
        print(f"\nRunning mode: {mode}")
        model = load_model(device)
        summary = run_one_pass(
            model=model,
            device=device,
            cached_batches=cached_batches,
            warmup_batches=args.warmup,
            measure_batches=args.measure,
            mode=mode,
        )
        results["modes"][mode] = summary

        print(
            f"[{mode}] total mean={summary['total_ms']['mean']:.2f}ms "
            f"fps={summary['fps_from_mean_total']:.2f} "
            f"geom={summary['geometry_ms']['mean']:.2f} cam={summary['camencode_ms']['mean']:.2f} "
            f"voxel={summary['voxel_pool_ms']['mean']:.2f} bev={summary['bevencode_ms']['mean']:.2f}"
        )

    if "torch" in results["modes"] and "trt" in results["modes"]:
        torch_fps = results["modes"]["torch"]["fps_from_mean_total"]
        trt_fps = results["modes"]["trt"]["fps_from_mean_total"]
        speedup = trt_fps / max(torch_fps, 1e-9)
        results["speedup_trt_over_torch_fps"] = speedup
        print(f"\nSpeedup (TRT over Torch, fps): {speedup:.2f}x")

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Saved benchmark report to: {output_path}")


if __name__ == "__main__":
    main()
