# LSS BEV Streaming Demo (Portfolio)

A real-time Bird's-Eye-View (BEV) segmentation demo built on Lift-Splat-Shoot (ECCV 2020), with:
- Multi-camera nuScenes input pipeline
- Live camera + BEV MJPEG streaming (`/cam`, `/bev`)
- Online LiDAR-referenced evaluation dashboard (`/eval`, `/eval.json`)
- Optional TensorRT acceleration for `camencode` and `bevencode`

## 1) What this repo contains

- `stream_inference.py` - thin compatibility entrypoint (`python stream_inference.py`)
- `src/streaming/` - modular runtime package:
  - `config.py` - runtime knobs (dataset paths, TensorRT flags, stream params)
  - `inference.py` - core inference loop and profiling logs
  - `tensorrt_utils.py` - TensorRT build/load + compatibility patch
  - `eval_metrics.py` - online LiDAR-referenced metrics and `/eval` rendering
  - `http_server.py` - `/cam`, `/bev`, `/eval`, `/eval.json` handlers
  - `visualization.py` - camera tiling, BEV coloring, overlays
  - `state.py` - thread-safe shared frame/eval state
- `src/` - minimal model/data utilities used by streaming runtime
- `requirements.txt` - Python dependencies

## 2) Prerequisites

- Python 3.8-3.11 (Python 3.12+ is not supported by current `nuscenes-devkit`/`Shapely<2` dependency chain)
- CUDA-capable GPU recommended
- nuScenes dataset available locally
- Trained LSS weights (`lss_clean_weights.pth`)

## 3) Setup

### Option A: Standard Python environment (portable)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If your system default `python3` is 3.12+ (for example 3.13), create the environment explicitly with 3.11 to avoid `Shapely` build failures.

If you see `OSError: ... libgeos_c.so ... no such file`, install GEOS in your active environment before running:

```bash
sudo apt-get install -y libgeos-dev
```

### Option B: Jetson/aarch64 with CUDA (recommended for GPU inference)

On Jetson, install a PyTorch/TorchVision build that matches your JetPack version (from NVIDIA wheels), then install this repo's dependencies.

Typical workflow:

```bash
python3.8 -m venv .venv-jp
source .venv-jp/bin/activate
pip install --upgrade pip setuptools wheel
# Install NVIDIA-provided torch/torchvision wheels for your JetPack here.
pip install -r requirements.txt
```

> Important: On some JetPack releases, CUDA-enabled PyTorch wheels are only available for specific Python versions (commonly Python 3.8 on JP5.x).

### Option C: Clone an existing known-good CUDA env (optional)

If you already have a working local env with CUDA-enabled PyTorch, you can clone it (this does not modify the source env):

```bash
conda create -y -n lss-bev-gpu --clone <existing_cuda_env_name_or_path>
conda activate lss-bev-gpu
pip install -r requirements.txt
```

## 4) Required assets

Default values expected by the streaming runtime:
- Dataset root: `/mnt/nvme/data/sets/nuscenes`
- Weights file: `lss_clean_weights.pth` (in repo root)

If needed, edit constants in `src/streaming/config.py`:
- `DATAROOT`
- `NUSCENES_VERSION`
- `WEIGHTS_PATH`
- `HOST_IP`, `PORT`

## 5) Run

```bash
python stream_inference.py
```

If using a Python 3.11 conda env and you hit `GLIBCXX_*` or loader issues, run with:

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python stream_inference.py
```

Open in browser:
- Camera stream: `http://<HOST_IP>:8080/cam`
- BEV stream: `http://<HOST_IP>:8080/bev`
- Eval dashboard: `http://<HOST_IP>:8080/eval`
- Eval JSON: `http://<HOST_IP>:8080/eval.json`

## 6) Optional TensorRT

The runtime can auto-build/load TensorRT engines when `torch2trt` is available.
Generated engine files (default):
- `lss_camencode_trt_engine.pth`
- `lss_bevencode_trt_engine.pth`

### How the ~20x speedup happens (camencode + bevencode)

The speedup does **not** come from replacing the entire pipeline with TensorRT. It comes from targeting the two heaviest convolutional stages and keeping shape-sensitive glue logic in PyTorch.

**Pipeline stages in this project**
- `get_geometry(...)` (PyTorch): camera calibration transforms
- `camencode(...)` (**TensorRT candidate**): per-camera CNN feature extraction
- `voxel_pooling(...)` (PyTorch): lift+splat aggregation into BEV volume
- `bevencode(...)` (**TensorRT candidate**): BEV CNN head for segmentation logits

In practice, `camencode` and `bevencode` dominate GPU compute time, while geometry and voxel pooling are smaller or harder to export cleanly. Converting just those two blocks captures most of the available acceleration with minimal behavior risk.

**Why these two blocks accelerate so much**
- TensorRT fuses Conv+BN+activation patterns into fewer kernels.
- TensorRT chooses optimized kernels per layer shape for your device.
- FP16 execution (`USE_FP16=True`) increases tensor core utilization and cuts memory bandwidth pressure.
- Reduced kernel launch overhead and better memory planning inside engine segments.
- Reusing serialized engines (`*.pth`) avoids rebuild cost on subsequent runs.

**What remains in PyTorch (and why)**
- Geometry projection and voxel pooling remain in PyTorch to preserve exact project-specific logic and avoid brittle graph export for dynamic indexing/scatter-like operations.
- This hybrid design is a common production pattern: accelerate dense CNN trunks/heads with TensorRT and keep custom data movement/geometry in PyTorch.
- Voxel pooling is still GPU-parallelized in PyTorch (vectorized tensor ops + scatter-style `index_put_(accumulate=True)`), so it is not a serial CPU fallback path.

**How to reproduce and verify in this repo**
1. Run baseline (PyTorch only): set `USE_TENSORRT = False` in `src/streaming/config.py`.
2. Run hybrid TRT mode: set `USE_TENSORRT = True`.
3. Compare steady-state logs from `inference.py`:
   - `fps=...`
   - `infer_total=...`
   - `geom=... cam=... voxel=... bev=...`
4. Confirm backend label changes:
   - `Torch`
   - `TRT(cam)` / `TRT(bev)` / `TRT(cam+bev)`

The expected signature of a successful optimization is:
- large drop in `cam` and `bev` stage times,
- small/no change in `geom` and `voxel`,
- large increase in end-to-end `fps`.

**Notes on first run vs warm run**
- First TensorRT run may be slower due to engine build + tactic selection.
- Subsequent runs load prebuilt engines and show true steady-state performance.
- The headline speedup should be reported from warm runs with matching input resolution/batch shape.

## 7) Credits

Core Lift-Splat-Shoot components are derived from NVIDIA's original project:
- Paper: Lift, Splat, Shoot (ECCV 2020)
- License and attribution retained in source headers and `LICENSE`.
