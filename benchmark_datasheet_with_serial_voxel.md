# Benchmark Datasheet: Torch vs TensorRT vs Serial Voxel

Generated: 2026-02-17 17:13:27

## Run Configuration

- Platform: `NVIDIA Jetson AGX Orin`
- Device: `cuda`
- Dataset root: `/mnt/nvme/data/sets/nuscenes`
- Weights: `/mnt/nvme/lss-bev-portfolio/lss_clean_weights.pth`
- Cached batches: `12`
- Warmup iterations per mode: `2`
- Measured iterations per mode: `6`

## FPS Comparison

| Mode | FPS | Total Mean Latency (ms) |
|---|---:|---:|
| `torch` | 14.32 | 69.82 |
| `trt` | 27.82 | 35.95 |
| `torch_serial_voxel` (unparallelized voxel reference) | 0.53 | 1883.38 |

- TRT over Torch speedup: **1.94x**
- Parallel voxel pooling over serial voxel pooling speedup: **26.98x**
- Combined speedup (`trt` over `torch_serial_voxel`): **52.49x**

### What exactly was optimized

1. **Voxel pooling path**:
   - `torch_serial_voxel` uses intentionally **unparallelized** reference accumulation (naive per-point loop), used here as the "out-of-the-box/unoptimized LSS-style voxel baseline" for comparison.
   - `torch` uses the production parallel PyTorch voxel path (`index_put_(accumulate=True)` scatter accumulation).
   - This step alone gives the large ~27x uplift.
2. **CNN encoder/head path**:
   - `trt` keeps the same parallel voxel path, but accelerates `camencode` + `bevencode` with TensorRT.
   - This gives an additional ~1.94x over the PyTorch-parallel baseline.

So the reported speedup is a stack: **voxel parallelization first**, then **TensorRT for cam/bev**.

## Stage Timing (Mean, ms)

| Mode | Geometry | CamEncode | Voxel Pool | BEV Encode | Total |
|---|---:|---:|---:|---:|---:|
| `torch` | 4.94 | 37.79 | 7.97 | 18.98 | 69.82 |
| `trt` | 4.43 | 12.17 | 8.96 | 10.27 | 35.95 |
| `torch_serial_voxel` | 5.03 | 38.92 | 1819.50 | 19.77 | 1883.38 |

## Raw Artifact

- JSON report: `benchmark_results_with_serial_voxel.json`
