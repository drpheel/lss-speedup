# LSS BEV Streaming Demo (Portfolio)

A real-time Bird's-Eye-View (BEV) segmentation demo built on Lift-Splat-Shoot (ECCV 2020), with:
- Multi-camera nuScenes input pipeline
- Live camera + BEV MJPEG streaming (`/cam`, `/bev`)
- Online LiDAR-referenced evaluation dashboard (`/eval`, `/eval.json`)
- Optional TensorRT acceleration for `camencode` and `bevencode`

## 1) What this repo contains

- `stream_inference.py` - real-time inference + HTTP streaming server
- `src/` - minimal model/data utilities required by the script
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

Default values expected by `stream_inference.py`:
- Dataset root: `/mnt/nvme/data/sets/nuscenes`
- Weights file: `lss_clean_weights.pth` (in repo root)

If needed, edit constants near the top of `stream_inference.py`:
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

The script can auto-build/load TensorRT engines when `torch2trt` is available.
Generated engine files (default):
- `lss_camencode_trt_engine.pth`
- `lss_bevencode_trt_engine.pth`

## 7) Credits

Core Lift-Splat-Shoot components are derived from NVIDIA's original project:
- Paper: Lift, Splat, Shoot (ECCV 2020)
- License and attribution retained in source headers and `LICENSE`.
