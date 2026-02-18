"""Configuration for live LSS streaming inference.

Values can be overridden from environment variables.
If a project-local `.env` file exists, it is loaded first.
"""

import os
from pathlib import Path
from typing import Tuple


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_REPO_ROOT = Path(__file__).resolve().parents[2]
_load_dotenv(_REPO_ROOT / ".env")

def _resolve_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


DATAROOT = _resolve_path(os.getenv("DATAROOT", "./data/sets/nuscenes"))
NUSCENES_VERSION = os.getenv("NUSCENES_VERSION", "mini")  # "mini" or "trainval"
WEIGHTS_PATH = _resolve_path(os.getenv("WEIGHTS_PATH", "lss_clean_weights.pth"))
HOST_IP = os.getenv("HOST_IP", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
STREAM_FPS = int(os.getenv("STREAM_FPS", "20"))
FRAME_INTERVAL = 1.0 / STREAM_FPS
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "70"))
STREAM_SCALE = float(os.getenv("STREAM_SCALE", "0.6"))
USE_FP16 = _env_bool("USE_FP16", True)
SHOW_OVERLAY = _env_bool("SHOW_OVERLAY", False)
LOG_EVERY_SEC = float(os.getenv("LOG_EVERY_SEC", "1.0"))
CACHE_BATCHES = int(os.getenv("CACHE_BATCHES", "16"))
LOOP_DATASET = _env_bool("LOOP_DATASET", True)
PREFETCH_FACTOR = int(os.getenv("PREFETCH_FACTOR", "2"))
USE_TENSORRT = _env_bool("USE_TENSORRT", True)
TRT_DEBUG_TRACEBACK = _env_bool("TRT_DEBUG_TRACEBACK", True)
ENABLE_LIDAR_BEV_EVAL = _env_bool("ENABLE_LIDAR_BEV_EVAL", True)

EVAL_THRESHOLD = 0.5
EVAL_DISTANCE_BINS_M: Tuple[Tuple[float, float], ...] = (
    (0.0, 10.0),
    (10.0, 20.0),
    (20.0, 30.0),
)
EVAL_NEARFIELD_RADIUS_M = 20.0
EVAL_NEARFIELD_WEIGHT = 3.0
EVAL_EGO_X_OFFSET_M = 0.0
EVAL_MIN_IOU = 0.45
EVAL_MIN_PRECISION = 0.75
EVAL_MIN_RECALL = 0.80
EVAL_MIN_F1 = 0.77
EVAL_MIN_NEARFIELD_WEIGHTED_IOU = 0.50
EVAL_BIN_MIN_IOU = {
    "0-10m": 0.60,
    "10-20m": 0.50,
    "20-30m": 0.45,
}

TENSORRT_CAM_ENGINE_PATH = _resolve_path(
    os.getenv("TENSORRT_CAM_ENGINE_PATH", "lss_camencode_trt_engine.pth")
)
TENSORRT_BEV_ENGINE_PATH = _resolve_path(
    os.getenv("TENSORRT_BEV_ENGINE_PATH", "lss_bevencode_trt_engine.pth")
)
TENSORRT_WORKSPACE_MB = int(os.getenv("TENSORRT_WORKSPACE_MB", "1024"))

EGO_LENGTH_M = 4.084
EGO_WIDTH_M = 1.85
EGO_X_OFFSET_M = 0.5

GRID_CONF = {
    "xbound": [-50.0, 50.0, 0.5],
    "ybound": [-50.0, 50.0, 0.5],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [4.0, 45.0, 1.0],
}

DATA_AUG_CONF = {
    "resize_lim": (0.193, 0.225),
    "final_dim": (128, 352),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "bot_pct_lim": (0.0, 0.22),
    "cams": [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ],
    "Ncams": 5,
}
