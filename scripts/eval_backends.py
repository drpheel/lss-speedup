#!/usr/bin/env python3
"""Evaluate BEV quality metrics for Torch vs TensorRT backends.

This script reuses the same model/data pipeline and RunningBevEval metrics
used by the streaming runtime, and writes reproducible artifacts.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from src.data import compile_data
from src.models import LiftSplatShoot
from src.streaming.config import (
    DATA_AUG_CONF,
    DATAROOT,
    EVAL_BIN_MIN_IOU,
    EVAL_MIN_F1,
    EVAL_MIN_IOU,
    EVAL_MIN_NEARFIELD_WEIGHTED_IOU,
    EVAL_MIN_PRECISION,
    EVAL_MIN_RECALL,
    EVAL_THRESHOLD,
    GRID_CONF,
    NUSCENES_VERSION,
    PREFETCH_FACTOR,
    TENSORRT_BEV_ENGINE_PATH,
    TENSORRT_CAM_ENGINE_PATH,
    USE_FP16,
    WEIGHTS_PATH,
)
from src.streaming.eval_metrics import RunningBevEval
from src.streaming.tensorrt_utils import HAS_TORCH2TRT, maybe_build_single_input_tensorrt


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
        prefetch_factor=PREFETCH_FACTOR,
    )

    cached = []
    for i, batch in enumerate(loader):
        cached.append(batch)
        if i + 1 >= num_batches:
            break
    if not cached:
        raise RuntimeError("No validation batches were cached; check DATAROOT and split.")
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


def _checks(eval_summary: Dict[str, object]) -> Dict[str, object]:
    near = eval_summary.get("nearfield_weighting", {})
    bins = eval_summary.get("distance_bins", {})

    global_checks = {
        "iou": {
            "value": float(eval_summary.get("iou", 0.0)),
            "threshold": EVAL_MIN_IOU,
            "pass": float(eval_summary.get("iou", 0.0)) >= EVAL_MIN_IOU,
        },
        "precision": {
            "value": float(eval_summary.get("precision", 0.0)),
            "threshold": EVAL_MIN_PRECISION,
            "pass": float(eval_summary.get("precision", 0.0)) >= EVAL_MIN_PRECISION,
        },
        "recall": {
            "value": float(eval_summary.get("recall", 0.0)),
            "threshold": EVAL_MIN_RECALL,
            "pass": float(eval_summary.get("recall", 0.0)) >= EVAL_MIN_RECALL,
        },
        "f1": {
            "value": float(eval_summary.get("f1", 0.0)),
            "threshold": EVAL_MIN_F1,
            "pass": float(eval_summary.get("f1", 0.0)) >= EVAL_MIN_F1,
        },
        "nearfield_weighted_iou": {
            "value": float(near.get("weighted_iou", 0.0)),
            "threshold": EVAL_MIN_NEARFIELD_WEIGHTED_IOU,
            "pass": float(near.get("weighted_iou", 0.0)) >= EVAL_MIN_NEARFIELD_WEIGHTED_IOU,
        },
    }

    bin_checks = {}
    for label, thr in EVAL_BIN_MIN_IOU.items():
        val = float(bins.get(label, {}).get("iou", 0.0))
        bin_checks[label] = {"value": val, "threshold": thr, "pass": val >= thr}

    overall_pass = all(item["pass"] for item in global_checks.values()) and all(
        item["pass"] for item in bin_checks.values()
    )
    return {
        "overall_pass": overall_pass,
        "global": global_checks,
        "bins": bin_checks,
    }


def run_mode(
    model: LiftSplatShoot,
    device: torch.device,
    cached_batches: List[tuple],
    warmup_iters: int,
    measure_iters: int,
    mode: str,
) -> Dict[str, object]:
    assert mode in {"torch", "trt"}
    use_trt = mode == "trt"
    if use_trt and device.type != "cuda":
        raise RuntimeError("TRT mode requires CUDA.")
    if use_trt and not HAS_TORCH2TRT:
        raise RuntimeError("TRT mode requested but torch2trt is unavailable.")

    camencode_runner = model.camencode
    bevencode_runner = model.bevencode
    using_trt_cam = False
    using_trt_bev = False

    # Build/load TRT engines once from representative shapes.
    first_batch = move_batch_to_device(cached_batches[0], device)
    imgs0, _, _, _, _, _, _ = first_batch
    if use_trt:
        sample_cam_input = imgs0.view(-1, imgs0.shape[2], imgs0.shape[3], imgs0.shape[4])
        camencode_runner, using_trt_cam = maybe_build_single_input_tensorrt(
            model.camencode,
            sample_cam_input,
            device,
            TENSORRT_CAM_ENGINE_PATH,
            "camencode",
        )

    tracker = RunningBevEval(threshold=EVAL_THRESHOLD)
    total_iters = warmup_iters + measure_iters
    t0 = time.time()
    for i in range(total_iters):
        batch = cached_batches[i % len(cached_batches)]
        imgs, rots, trans, intrinsics, post_rots, post_trans, binimgs = move_batch_to_device(
            batch, device
        )

        with torch.inference_mode():
            with torch.cuda.amp.autocast(
                enabled=(USE_FP16 and device.type == "cuda" and not use_trt)
            ):
                bsz, ncams, ch, im_h, im_w = imgs.shape
                geom = model.get_geometry(rots, trans, intrinsics, post_rots, post_trans)
                cam_in = imgs.view(bsz * ncams, ch, im_h, im_w)
                cam_feats = camencode_runner(cam_in) if using_trt_cam else model.camencode(cam_in)
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

                if use_trt and not using_trt_bev:
                    bevencode_runner, using_trt_bev = maybe_build_single_input_tensorrt(
                        model.bevencode,
                        vox,
                        device,
                        TENSORRT_BEV_ENGINE_PATH,
                        "bevencode",
                    )

                bev_logits = bevencode_runner(vox) if using_trt_bev else model.bevencode(vox)

        if i >= warmup_iters:
            tracker.update(bev_logits, binimgs)

    elapsed = time.time() - t0
    summary = tracker.summary()
    summary["threshold_checks"] = _checks(summary)
    summary["backend_mode"] = mode
    summary["using_trt_camencode"] = using_trt_cam
    summary["using_trt_bevencode"] = using_trt_bev
    summary["elapsed_s"] = elapsed
    return summary


def _write_markdown(path: Path, results: Dict[str, object]) -> None:
    modes = results.get("modes", {})
    lines = [
        "# Eval Quality Report",
        "",
        f"- Device: `{results.get('device')}` ({results.get('cuda_device_name')})",
        f"- Dataroot: `{results.get('dataroot')}`",
        f"- Cached batches: `{results.get('num_cached_batches')}`",
        f"- Warmup iters: `{results.get('warmup_iters')}`",
        f"- Measure iters: `{results.get('measure_iters')}`",
        "",
        "## Global Metrics",
        "",
        "| Mode | IoU | Precision | Recall | F1 | Near-weighted IoU | Overall Pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, m in modes.items():
        checks = m.get("threshold_checks", {})
        lines.append(
            "| `{}` | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
                mode,
                m.get("iou", 0.0),
                m.get("precision", 0.0),
                m.get("recall", 0.0),
                m.get("f1", 0.0),
                m.get("nearfield_weighting", {}).get("weighted_iou", 0.0),
                "PASS" if checks.get("overall_pass", False) else "FAIL",
            )
        )
    lines += [
        "",
        "## Distance Bin IoU",
        "",
        "| Mode | 0-10m | 10-20m | 20-30m |",
        "|---|---:|---:|---:|",
    ]
    for mode, m in modes.items():
        bins = m.get("distance_bins", {})
        lines.append(
            "| `{}` | {:.3f} | {:.3f} | {:.3f} |".format(
                mode,
                bins.get("0-10m", {}).get("iou", 0.0),
                bins.get("10-20m", {}).get("iou", 0.0),
                bins.get("20-30m", {}).get("iou", 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Torch/TRT BEV quality metrics.")
    parser.add_argument("--num-batches", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--measure", type=int, default=24)
    parser.add_argument("--modes", type=str, default="torch,trt")
    parser.add_argument("--output", type=str, default="eval_quality_results.json")
    parser.add_argument("--output-md", type=str, default="eval_quality_report.md")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in {"torch", "trt"}:
            raise ValueError(f"Unsupported mode: {m}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"Device: {device} ({cuda_name})")

    cached_batches = build_cached_batches(device, args.num_batches)

    out = {
        "timestamp_unix": time.time(),
        "device": str(device),
        "cuda_device_name": cuda_name,
        "dataroot": DATAROOT,
        "weights_path": WEIGHTS_PATH,
        "num_cached_batches": len(cached_batches),
        "warmup_iters": args.warmup,
        "measure_iters": args.measure,
        "modes": {},
    }

    for mode in modes:
        print(f"\nRunning eval mode: {mode}")
        model = load_model(device)
        summary = run_mode(
            model=model,
            device=device,
            cached_batches=cached_batches,
            warmup_iters=args.warmup,
            measure_iters=args.measure,
            mode=mode,
        )
        out["modes"][mode] = summary
        checks = summary.get("threshold_checks", {})
        print(
            f"[{mode}] IoU={summary['iou']:.3f} P={summary['precision']:.3f} "
            f"R={summary['recall']:.3f} F1={summary['f1']:.3f} "
            f"near_w_iou={summary['nearfield_weighting']['weighted_iou']:.3f} "
            f"overall={'PASS' if checks.get('overall_pass', False) else 'FAIL'}"
        )

    if "torch" in out["modes"] and "trt" in out["modes"]:
        out["delta_trt_minus_torch"] = {
            "iou": out["modes"]["trt"]["iou"] - out["modes"]["torch"]["iou"],
            "precision": out["modes"]["trt"]["precision"] - out["modes"]["torch"]["precision"],
            "recall": out["modes"]["trt"]["recall"] - out["modes"]["torch"]["recall"],
            "f1": out["modes"]["trt"]["f1"] - out["modes"]["torch"]["f1"],
            "near_weighted_iou": (
                out["modes"]["trt"]["nearfield_weighting"]["weighted_iou"]
                - out["modes"]["torch"]["nearfield_weighting"]["weighted_iou"]
            ),
        }

    out_json = Path(args.output)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved eval JSON: {out_json}")

    out_md = Path(args.output_md)
    _write_markdown(out_md, out)
    print(f"Saved eval report: {out_md}")


if __name__ == "__main__":
    main()

