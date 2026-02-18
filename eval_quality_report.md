# Eval Quality Report

- Device: `cuda` (Orin)
- Dataroot: `/mnt/nvme/data/sets/nuscenes`
- Cached batches: `24`
- Warmup iters: `4`
- Measure iters: `24`

## Global Metrics

| Mode | IoU | Precision | Recall | F1 | Near-weighted IoU | Overall Pass |
|---|---:|---:|---:|---:|---:|---:|
| `torch` | 0.371 | 0.543 | 0.540 | 0.541 | 0.437 | FAIL |
| `trt` | 0.371 | 0.542 | 0.540 | 0.541 | 0.436 | FAIL |

## Distance Bin IoU

| Mode | 0-10m | 10-20m | 20-30m |
|---|---:|---:|---:|
| `torch` | 0.628 | 0.511 | 0.409 |
| `trt` | 0.628 | 0.510 | 0.409 |
