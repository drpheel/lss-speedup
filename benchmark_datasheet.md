# Benchmark Datasheet: Speedup Breakdown (Voxel + TensorRT)

Generated: 2026-02-17 17:13:27

## Run Configuration

- Device: `cuda`
- Dataset root: `/mnt/nvme/data/sets/nuscenes`
- Weights: `/mnt/nvme/lss-bev-portfolio/lss_clean_weights.pth`
- Cached batches: `12`
- Warmup iterations per mode: `2`
- Measured iterations per mode: `6`

## FPS Summary (Including Unparallelized Voxel Baseline)

| Mode | FPS | Total Mean Latency (ms) |
|---|---:|---:|
| `torch_serial_voxel` (unparallelized voxel reference; out-of-the-box/unoptimized LSS-style baseline) | 0.53 | 1883.38 |
| `torch` (parallel voxel pooling) | 14.32 | 69.82 |
| `trt` (parallel voxel + TRT cam/bev) | 27.82 | 35.95 |

## Speedup Interpretation

- **Voxel pooling parallelization only** (`torch` vs `torch_serial_voxel`): **26.98x FPS**
- **TensorRT cam/bev only** (on top of parallel voxel path, `trt` vs `torch`): **1.94x FPS**
- **Combined vs unparallelized baseline** (`trt` vs `torch_serial_voxel`): **52.49x FPS**

This makes the optimization stack explicit:
1. Move from serial voxel accumulation to parallel scatter accumulation in PyTorch.
2. Then offload `camencode` and `bevencode` to TensorRT.

## Stage Timing (Mean, ms)

| Mode | Geometry | CamEncode | Voxel Pool | BEV Encode | Total |
|---|---:|---:|---:|---:|---:|
| `torch_serial_voxel` | 5.03 | 38.92 | 1819.50 | 19.77 | 1883.38 |
| `torch` | 4.94 | 37.79 | 7.97 | 18.98 | 69.82 |
| `trt` | 4.43 | 12.17 | 8.96 | 10.27 | 35.95 |

The dominant difference between unparallelized and parallel runs is voxel pooling time (1819.50 ms -> 7.97 ms). The dominant difference between `torch` and `trt` runs is `camencode`/`bevencode`.

## Raw Artifact

- JSON report: `benchmark_results_with_serial_voxel.json`
