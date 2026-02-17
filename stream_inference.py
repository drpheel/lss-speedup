import time
import torch
import cv2
import numpy as np
import threading
import os
import traceback
import importlib
import json
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from src.models import LiftSplatShoot
from src.data import compile_data 

try:
    from torch2trt import torch2trt, TRTModule
    HAS_TORCH2TRT = True
except Exception:
    HAS_TORCH2TRT = False

_TORCH2TRT_PATCHED = False

# --- CONFIGURATION ---
DATAROOT = "/mnt/nvme/data/sets/nuscenes"
NUSCENES_VERSION = "mini"  # "mini" or "trainval"
WEIGHTS_PATH = "lss_clean_weights.pth"
HOST_IP = "0.0.0.0" 
PORT = 8080
STREAM_FPS = 20
FRAME_INTERVAL = 1.0 / STREAM_FPS
JPEG_QUALITY = 70
STREAM_SCALE = 0.6
USE_FP16 = True
SHOW_OVERLAY = False
LOG_EVERY_SEC = 1.0
CACHE_BATCHES = 16  # set to 0 to iterate full val split without replay cache
LOOP_DATASET = True  # when False, stop after one full pass through val split
PREFETCH_FACTOR = 2
USE_TENSORRT = True
TRT_DEBUG_TRACEBACK = True
ENABLE_LIDAR_BEV_EVAL = True
EVAL_THRESHOLD = 0.5
EVAL_DISTANCE_BINS_M = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0))
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
TENSORRT_CAM_ENGINE_PATH = "lss_camencode_trt_engine.pth"
TENSORRT_BEV_ENGINE_PATH = "lss_bevencode_trt_engine.pth"
TENSORRT_WORKSPACE_MB = 1024
EGO_LENGTH_M = 4.084
EGO_WIDTH_M = 1.85
EGO_X_OFFSET_M = 0.5

# Globals to share frames between threads
global_cam_jpeg = None
global_bev_jpeg = None
frame_lock = threading.Lock()
eval_lock = threading.Lock()
global_eval_stats = {}


def _fmt_pct(x):
    return f"{100.0 * float(x):.1f}%"


def _fmt_num(x, nd=3):
    return f"{float(x):.{nd}f}"


def _status_class(value, threshold):
    if value is None or threshold is None:
        return "neutral"
    return "pass" if float(value) >= float(threshold) else "fail"


def _metric_cell(name, value, threshold=None):
    cls = _status_class(value, threshold)
    value_txt = _fmt_num(value) if value is not None else "-"
    th_txt = _fmt_num(threshold) if threshold is not None else "-"
    return (
        f"<tr class='{cls}'>"
        f"<td>{escape(name)}</td>"
        f"<td>{value_txt}</td>"
        f"<td>{th_txt}</td>"
        f"<td>{'OK' if cls == 'pass' else ('LOW' if cls == 'fail' else '-')}</td>"
        f"</tr>"
    )


def render_eval_html(metrics):
    if not metrics or not metrics.get("ready", False):
        msg = "Eval is warming up. Open /cam and /bev to verify streams are active."
        if metrics and "message" in metrics:
            msg = str(metrics["message"])
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="1">
<title>LSS Eval</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 16px; background: #121212; color: #e6e6e6; }}
.card {{ background: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 12px; max-width: 980px; }}
.hint {{ color: #f5c842; }}
a {{ color: #6aa9ff; }}
</style></head><body>
<div class="card">
<h2>BEV Eval (LiDAR Reference)</h2>
<p class="hint">{escape(msg)}</p>
<p><a href="/eval.json">Raw JSON</a></p>
</div>
</body></html>"""

    near = metrics.get("nearfield_weighting", {})
    bins = metrics.get("distance_bins", {})
    backend = escape(str(metrics.get("backend", "unknown")))
    frames = int(metrics.get("frames", 0))
    updated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(metrics.get("timestamp_unix", time.time()))))

    global_rows = "".join([
        _metric_cell("IoU", metrics.get("iou"), EVAL_MIN_IOU),
        _metric_cell("Precision", metrics.get("precision"), EVAL_MIN_PRECISION),
        _metric_cell("Recall", metrics.get("recall"), EVAL_MIN_RECALL),
        _metric_cell("F1", metrics.get("f1"), EVAL_MIN_F1),
        _metric_cell("Near-field weighted IoU", near.get("weighted_iou"), EVAL_MIN_NEARFIELD_WEIGHTED_IOU),
        _metric_cell("Pos rate diff (|pred-tgt|)", abs(metrics.get("pos_rate_diff", 0.0)), 0.08),
    ])

    bin_rows = []
    for label, vals in bins.items():
        thr = EVAL_BIN_MIN_IOU.get(label)
        bin_rows.append(_metric_cell(f"{label} IoU", vals.get("iou"), thr))
    bins_html = "".join(bin_rows) if bin_rows else "<tr class='neutral'><td colspan='4'>No bin stats yet</td></tr>"

    near_radius = _fmt_num(near.get("radius_m", EVAL_NEARFIELD_RADIUS_M), nd=1)
    near_weight = _fmt_num(near.get("weight", EVAL_NEARFIELD_WEIGHT), nd=1)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="1">
  <title>LSS BEV Eval</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; background: #111; color: #eee; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; }}
    .card {{ background: #1b1b1b; border: 1px solid #333; border-radius: 8px; padding: 12px; }}
    .meta {{ color: #c5c5c5; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border: 1px solid #3a3a3a; padding: 8px; text-align: left; }}
    th {{ background: #262626; }}
    tr.pass td {{ background: rgba(29, 185, 84, 0.20); color: #9ef3b9; }}
    tr.fail td {{ background: rgba(255, 69, 58, 0.24); color: #ffb3ad; }}
    tr.neutral td {{ background: rgba(120, 120, 120, 0.12); color: #ddd; }}
    a {{ color: #6aa9ff; }}
  </style>
</head>
<body>
  <h2>BEV Eval Dashboard (LiDAR Reference)</h2>
  <p class="meta">
    Backend: <b>{backend}</b> |
    Frames: <b>{frames}</b> |
    Updated: <b>{updated}</b> |
    Near-field config: <b>radius={near_radius}m, weight={near_weight}x</b> |
    <a href="/eval.json">Raw JSON</a>
  </p>
  <div class="grid">
    <div class="card">
      <h3>Distance-Binned IoU</h3>
      <table>
        <thead><tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th></tr></thead>
        <tbody>{bins_html}</tbody>
      </table>
      <p class="meta">Bins are radial distance from ego center in BEV.</p>
    </div>
  </div>
</body>
</html>"""

# --- MODEL SETUP ---
grid_conf = {
    'xbound': [-50.0, 50.0, 0.5], 'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0], 'dbound': [4.0, 45.0, 1.0],
}
data_aug_conf = {
    'resize_lim': (0.193, 0.225), 'final_dim': (128, 352),
    'rot_lim': (-5.4, 5.4), 'H': 900, 'W': 1600,
    'rand_flip': True, 'bot_pct_lim': (0.0, 0.22),
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 5,
}

# --- HELPER FUNCTIONS ---
def denormalize_img(img):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = img.transpose(1, 2, 0)
    img = (img * std + mean) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

def tile_cameras(imgs_tensor):
    imgs_np = [denormalize_img(img.cpu().numpy()) for img in imgs_tensor]
    top_row = np.hstack(imgs_np[:3])
    bot_row = np.hstack(imgs_np[3:])
    full_grid = np.vstack((top_row, bot_row))
    return cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR)


def draw_overlay(frame, lines):
    y = 24
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        y += 22


def maybe_resize(frame):
    if STREAM_SCALE >= 0.999:
        return frame
    h, w = frame.shape[:2]
    nw = max(1, int(w * STREAM_SCALE))
    nh = max(1, int(h * STREAM_SCALE))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def draw_ego_triangle(frame, grid_conf):
    h, w = frame.shape[:2]
    x_min, x_max, x_step = grid_conf["xbound"]
    y_min, y_max, y_step = grid_conf["ybound"]

    nx = max(1, int(round((x_max - x_min) / x_step)))
    ny = max(1, int(round((y_max - y_min) / y_step)))

    ego_x_idx = (EGO_X_OFFSET_M - x_min) / x_step
    ego_y_idx = (0.0 - y_min) / y_step
    cy = int(round((ego_x_idx / max(1, nx - 1)) * (h - 1)))
    cx = int(round((ego_y_idx / max(1, ny - 1)) * (w - 1)))
    cy = int(np.clip(cy, 0, h - 1))
    cx = int(np.clip(cx, 0, w - 1))

    # Keep marker physically scaled to BEV cell size after any output resize.
    row_scale = h / float(nx)
    col_scale = w / float(ny)
    tri_len = max(6, int(round((EGO_LENGTH_M / x_step) * row_scale)))
    tri_half_w = max(3, int(round((EGO_WIDTH_M / y_step) * col_scale * 0.5)))

    tip_y = min(h - 1, cy + tri_len // 2)
    base_y = max(0, cy - tri_len // 2)
    pts = np.array(
        [
            [cx, tip_y],                 # tip (forward / +x)
            [cx - tri_half_w, base_y],   # left rear
            [cx + tri_half_w, base_y],   # right rear
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], (255, 255, 255))
    cv2.polylines(frame, [pts], True, (0, 0, 0), 2, cv2.LINE_AA)


def maybe_build_single_input_tensorrt(module, sample_input, device, engine_path, label):
    if device.type != "cuda":
        print("TensorRT disabled: CUDA device not available.")
        return module, False
    if not HAS_TORCH2TRT:
        print("TensorRT disabled: torch2trt is not installed. Falling back to PyTorch.")
        return module, False
    _apply_torch2trt_conv_dims_patch()

    if os.path.exists(engine_path):
        try:
            trt_model = TRTModule()
            trt_model.load_state_dict(torch.load(engine_path))
            trt_model.to(device)
            trt_model.eval()
            print(f"Loaded {label} TensorRT engine from {engine_path}")
            return trt_model, True
        except Exception as exc:
            print(f"Failed to load {label} TensorRT engine, rebuilding. Reason: {exc}")
            if TRT_DEBUG_TRACEBACK:
                traceback.print_exc()

    print(f"Building {label} TensorRT engine. This can take a while on first run...")
    t0 = time.perf_counter()
    try:
        trt_model = torch2trt(
            module,
            [sample_input],
            fp16_mode=USE_FP16,
            max_workspace_size=TENSORRT_WORKSPACE_MB * (1 << 20),
        )
        torch.save(trt_model.state_dict(), engine_path)
        dt = time.perf_counter() - t0
        print(f"{label} TensorRT engine built and saved to {engine_path} in {dt:.1f}s")
        return trt_model, True
    except Exception as exc:
        print(f"{label} TensorRT build failed; falling back to PyTorch. Reason: {exc}")
        if TRT_DEBUG_TRACEBACK:
            traceback.print_exc()
        return module, False


class RunningBevEval:
    """Tracks running segmentation metrics against LiDAR-derived BEV targets."""
    def __init__(self, threshold=0.5):
        self.threshold = float(threshold)
        self.distance_bins = tuple((float(lo), float(hi)) for lo, hi in EVAL_DISTANCE_BINS_M)
        self.nearfield_radius_m = float(EVAL_NEARFIELD_RADIUS_M)
        self.nearfield_weight = float(EVAL_NEARFIELD_WEIGHT)
        self._cached_maps = {}
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.frames = 0
        self.last_iou = 0.0
        self.last_precision = 0.0
        self.last_recall = 0.0
        self.last_f1 = 0.0
        self.last_pos_rate_diff = 0.0
        self.wtp = 0.0
        self.wfp = 0.0
        self.wfn = 0.0
        self.bin_stats = {
            self._bin_label(lo, hi): {"tp": 0.0, "fp": 0.0, "fn": 0.0}
            for lo, hi in self.distance_bins
        }

    @staticmethod
    def _bin_label(lo, hi):
        return f"{int(lo)}-{int(hi)}m"

    def _build_distance_maps(self, h, w, device):
        key = (str(device), int(h), int(w))
        if key in self._cached_maps:
            return self._cached_maps[key]

        x_min, x_max = grid_conf["xbound"][:2]
        y_min, y_max = grid_conf["ybound"][:2]
        x_step = (x_max - x_min) / float(h)
        y_step = (y_max - y_min) / float(w)
        x_centers = x_min + (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * x_step
        y_centers = y_min + (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * y_step
        xx, yy = torch.meshgrid(x_centers, y_centers, indexing="ij")
        dist = torch.sqrt((xx - EVAL_EGO_X_OFFSET_M) ** 2 + (yy - 0.0) ** 2)
        dist = dist.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        bin_masks = []
        for lo, hi in self.distance_bins:
            mask = ((dist >= lo) & (dist < hi))
            bin_masks.append(mask)
        near_mask = dist <= self.nearfield_radius_m
        weights = torch.ones_like(dist, dtype=torch.float32)
        weights = torch.where(near_mask, torch.full_like(weights, self.nearfield_weight), weights)

        maps = {"bin_masks": bin_masks, "weights": weights}
        self._cached_maps[key] = maps
        return maps

    def update(self, bev_logits, bev_target):
        if bev_logits is None or bev_target is None:
            return
        with torch.no_grad():
            pred = (torch.sigmoid(bev_logits) >= self.threshold)
            tgt = (bev_target > 0.5)
            _, _, h, w = pred.shape
            maps = self._build_distance_maps(h, w, pred.device)
            weights = maps["weights"]
            tp = (pred & tgt).sum().item()
            fp = (pred & (~tgt)).sum().item()
            fn = ((~pred) & tgt).sum().item()
            tn = ((~pred) & (~tgt)).sum().item()
            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.tn += tn
            self.frames += int(bev_logits.shape[0])

            # Per-frame instantaneous values for quick drift spotting.
            inter = tp
            union = tp + fp + fn
            self.last_iou = inter / max(union, 1.0)
            self.last_precision = tp / max(tp + fp, 1.0)
            self.last_recall = tp / max(tp + fn, 1.0)
            self.last_f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
            pred_pos_rate = (tp + fp) / max(tp + fp + fn + tn, 1.0)
            tgt_pos_rate = (tp + fn) / max(tp + fp + fn + tn, 1.0)
            self.last_pos_rate_diff = pred_pos_rate - tgt_pos_rate

            pred_f = pred.float()
            tgt_f = tgt.float()
            self.wtp += (pred_f * tgt_f * weights).sum().item()
            self.wfp += (pred_f * (1.0 - tgt_f) * weights).sum().item()
            self.wfn += ((1.0 - pred_f) * tgt_f * weights).sum().item()

            for (lo, hi), mask in zip(self.distance_bins, maps["bin_masks"]):
                label = self._bin_label(lo, hi)
                self.bin_stats[label]["tp"] += (pred & tgt & mask).sum().item()
                self.bin_stats[label]["fp"] += (pred & (~tgt) & mask).sum().item()
                self.bin_stats[label]["fn"] += ((~pred) & tgt & mask).sum().item()

    def summary(self):
        iou = self.tp / max(self.tp + self.fp + self.fn, 1.0)
        precision = self.tp / max(self.tp + self.fp, 1.0)
        recall = self.tp / max(self.tp + self.fn, 1.0)
        f1 = 2.0 * self.tp / max(2.0 * self.tp + self.fp + self.fn, 1.0)
        acc = (self.tp + self.tn) / max(self.tp + self.fp + self.fn + self.tn, 1.0)
        pred_pos_rate = (self.tp + self.fp) / max(self.tp + self.fp + self.fn + self.tn, 1.0)
        tgt_pos_rate = (self.tp + self.fn) / max(self.tp + self.fp + self.fn + self.tn, 1.0)
        weighted_iou = self.wtp / max(self.wtp + self.wfp + self.wfn, 1.0)
        weighted_precision = self.wtp / max(self.wtp + self.wfp, 1.0)
        weighted_recall = self.wtp / max(self.wtp + self.wfn, 1.0)
        weighted_f1 = 2.0 * self.wtp / max(2.0 * self.wtp + self.wfp + self.wfn, 1.0)
        bin_summary = {}
        for label, s in self.bin_stats.items():
            bt, bf, bn = s["tp"], s["fp"], s["fn"]
            bin_summary[label] = {
                "iou": float(bt / max(bt + bf + bn, 1.0)),
                "precision": float(bt / max(bt + bf, 1.0)),
                "recall": float(bt / max(bt + bn, 1.0)),
                "f1": float(2.0 * bt / max(2.0 * bt + bf + bn, 1.0)),
            }
        return {
            "frames": int(self.frames),
            "threshold": self.threshold,
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(acc),
            "pred_pos_rate": float(pred_pos_rate),
            "tgt_pos_rate": float(tgt_pos_rate),
            "pos_rate_diff": float(pred_pos_rate - tgt_pos_rate),
            "nearfield_weighting": {
                "radius_m": self.nearfield_radius_m,
                "weight": self.nearfield_weight,
                "weighted_iou": float(weighted_iou),
                "weighted_precision": float(weighted_precision),
                "weighted_recall": float(weighted_recall),
                "weighted_f1": float(weighted_f1),
            },
            "distance_bins": bin_summary,
            "last_iou": float(self.last_iou),
            "last_precision": float(self.last_precision),
            "last_recall": float(self.last_recall),
            "last_f1": float(self.last_f1),
            "last_pos_rate_diff": float(self.last_pos_rate_diff),
        }


def _apply_torch2trt_conv_dims_patch():
    global _TORCH2TRT_PATCHED
    if _TORCH2TRT_PATCHED or not HAS_TORCH2TRT:
        return
    try:
        import tensorrt as trt
        t2t = importlib.import_module('torch2trt.torch2trt')
        nc = importlib.import_module('torch2trt.converters.native_converters')

        def _flatten_ints(v):
            if isinstance(v, (list, tuple)):
                out = []
                for item in v:
                    out.extend(_flatten_ints(item))
                return out
            try:
                return [int(v)]
            except Exception:
                return [int(float(v))]

        def _normalize_nd_param(v, ndim, default):
            vals = _flatten_ints(v)
            if len(vals) == 0:
                vals = [int(default)]
            if len(vals) == 1:
                return tuple([vals[0]] * ndim)
            if len(vals) >= ndim:
                return tuple(vals[:ndim])
            return tuple(vals + [vals[-1]] * (ndim - len(vals)))

        def _to_trt_dims(vals):
            vals = _flatten_ints(vals)
            if len(vals) == 1:
                return trt.Dims((vals[0],))
            if len(vals) == 2 and hasattr(trt, "DimsHW"):
                return trt.DimsHW(vals[0], vals[1])
            return trt.Dims(tuple(vals))

        def convert_conv2d3d_patched(ctx):
            input = nc.get_arg(ctx, 'input', pos=0, default=None)
            weight = nc.get_arg(ctx, 'weight', pos=1, default=None)
            bias = nc.get_arg(ctx, 'bias', pos=2, default=None)
            stride = nc.get_arg(ctx, 'stride', pos=3, default=1)
            padding = nc.get_arg(ctx, 'padding', pos=4, default=0)
            dilation = nc.get_arg(ctx, 'dilation', pos=5, default=1)
            groups = nc.get_arg(ctx, 'groups', pos=6, default=1)

            input_trt = nc.add_missing_trt_tensors(ctx.network, [input])[0]
            output = ctx.method_return
            input_dim = input.dim() - 2

            out_channels = int(weight.shape[0])
            kernel_size = _normalize_nd_param(tuple(weight.shape[2:]), input_dim, default=1)
            stride = _normalize_nd_param(stride, input_dim, default=1)
            padding = _normalize_nd_param(padding, input_dim, default=0)
            dilation = _normalize_nd_param(dilation, input_dim, default=1)

            kernel = weight.detach().cpu().numpy()
            bias_np = bias.detach().cpu().numpy() if bias is not None else None

            if input_dim == 1:
                kernel_size = kernel_size + (1,)
                stride = stride + (1,)
                padding = padding + (0,)
                dilation = dilation + (1,)
                unsqueeze_layer = ctx.network.add_shuffle(input_trt)
                nc.set_layer_precision(ctx, unsqueeze_layer)
                unsqueeze_layer.reshape_dims = tuple([0] * input.ndim) + (1,)
                conv_input = unsqueeze_layer.get_output(0)
            else:
                conv_input = input_trt

            conv_layer = ctx.network.add_convolution_nd(
                input=conv_input,
                num_output_maps=out_channels,
                kernel_shape=kernel_size,
                kernel=kernel,
                bias=bias_np,
            )
            conv_layer.stride_nd = _to_trt_dims(stride)
            conv_layer.padding_nd = _to_trt_dims(padding)
            conv_layer.dilation_nd = _to_trt_dims(dilation)
            if groups is not None:
                conv_layer.num_groups = groups

            if input_dim == 1:
                squeeze_layer = ctx.network.add_shuffle(conv_layer.get_output(0))
                nc.set_layer_precision(ctx, squeeze_layer)
                squeeze_layer.reshape_dims = tuple([0] * input.ndim)
                output._trt = squeeze_layer.get_output(0)
            else:
                output._trt = conv_layer.get_output(0)

        for key in (
            'torch.nn.functional.conv1d',
            'torch.nn.functional.conv2d',
            'torch.nn.functional.conv3d',
        ):
            if key in t2t.CONVERTERS:
                t2t.CONVERTERS[key]['converter'] = convert_conv2d3d_patched
        _TORCH2TRT_PATCHED = True
        print("Applied torch2trt conv dims compatibility patch for TensorRT 8.5.")
    except Exception as exc:
        print(f"Could not apply torch2trt compatibility patch: {exc}")

# --- HTTP SERVER HANDLER ---
class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Route 1: Camera Stream
        if self.path == '/cam':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                with frame_lock:
                    frame_data = global_cam_jpeg
                if frame_data is None:
                    time.sleep(0.01)
                    continue
                
                try:
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame_data))
                    self.end_headers()
                    self.wfile.write(frame_data)
                    self.wfile.write(b'\r\n')
                    time.sleep(FRAME_INTERVAL)
                except BrokenPipeError:
                    break

        # Route 3: Eval metrics dashboard (human-readable)
        elif self.path == '/eval':
            with eval_lock:
                metrics = dict(global_eval_stats) if global_eval_stats else {
                    "ready": False,
                    "message": "Eval not started yet."
                }
            body = render_eval_html(metrics).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # Route 4: Eval metrics JSON (machine-readable)
        elif self.path == '/eval.json':
            with eval_lock:
                metrics = dict(global_eval_stats) if global_eval_stats else {
                    "ready": False,
                    "message": "Eval not started yet."
                }
            body = json.dumps(metrics, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # Route 2: BEV Stream
        elif self.path == '/bev':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                with frame_lock:
                    frame_data = global_bev_jpeg
                if frame_data is None:
                    time.sleep(0.01)
                    continue

                try:
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame_data))
                    self.end_headers()
                    self.wfile.write(frame_data)
                    self.wfile.write(b'\r\n')
                    time.sleep(FRAME_INTERVAL)
                except BrokenPipeError:
                    break
        else:
            self.send_error(404)
            self.end_headers()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

# --- INFERENCE LOOP ---
def inference_loop():
    global global_cam_jpeg, global_bev_jpeg, global_eval_stats
    
    print("Initializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = LiftSplatShoot(grid_conf, data_aug_conf, outC=1)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    model.to(device)
    model.eval()
    if hasattr(model, "profile_voxel_pooling"):
        model.profile_voxel_pooling = False

    print(f"Loading nuScenes {NUSCENES_VERSION} split from {DATAROOT}...")
    _, val_loader = compile_data(NUSCENES_VERSION, DATAROOT, data_aug_conf, grid_conf,
                                 bsz=1, nworkers=2, parser_name='segmentationdata')
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

    cached_batches = []
    if CACHE_BATCHES > 0:
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

    print("Inference Loop Started.")
    fps_ema = 0.0
    enc_ms_ema = 0.0
    fetch_ms_ema = 0.0
    copy_ms_ema = 0.0
    cached_idx = 0
    val_iter = iter(val_loader)
    last_log_t = time.perf_counter()
    camencode_runner = model.camencode
    bevencode_runner = model.bevencode
    using_tensorrt_camencode = False
    using_tensorrt_bevencode = False
    trt_cam_attempted = False
    trt_bev_attempted = False
    eval_tracker = RunningBevEval(threshold=EVAL_THRESHOLD) if ENABLE_LIDAR_BEV_EVAL else None
    while True:
        if cached_batches:
            imgs, rots, trans, intrinsics, post_rots, post_trans, binimgs = cached_batches[cached_idx]
            cached_idx = (cached_idx + 1) % len(cached_batches)
            fetch_ms = 0.0
        else:
            t_fetch0 = time.perf_counter()
            try:
                imgs, rots, trans, intrinsics, post_rots, post_trans, binimgs = next(val_iter)
            except StopIteration:
                if not LOOP_DATASET:
                    print("Reached end of validation split (single-pass mode).")
                    break
                val_iter = iter(val_loader)
                imgs, rots, trans, intrinsics, post_rots, post_trans, binimgs = next(val_iter)
            fetch_ms = (time.perf_counter() - t_fetch0) * 1000.0
        fetch_ms_ema = fetch_ms if fetch_ms_ema == 0.0 else (0.9 * fetch_ms_ema + 0.1 * fetch_ms)

        loop_t0 = time.perf_counter()
        imgs_cpu = imgs[0]
        should_log = (time.perf_counter() - last_log_t) >= LOG_EVERY_SEC

        # GPU Move
        t_h2d0 = time.perf_counter()
        imgs = imgs.to(device, non_blocking=True)
        rots = rots.to(device, non_blocking=True)
        trans = trans.to(device, non_blocking=True)
        intrinsics = intrinsics.to(device, non_blocking=True)
        post_rots = post_rots.to(device, non_blocking=True)
        post_trans = post_trans.to(device, non_blocking=True)
        t_h2d_ms = (time.perf_counter() - t_h2d0) * 1000.0

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
                # Geometry + pooling stay in PyTorch; cam/bev encoders can be independently TRT-accelerated.
                B, N, C, imH, imW = imgs.shape
                if should_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    stage_t0 = time.perf_counter()
                geom = model.get_geometry(rots, trans, intrinsics, post_rots, post_trans)
                if should_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    stage_t1 = time.perf_counter()

                cam_in = imgs.view(B * N, C, imH, imW)
                cam_feats = camencode_runner(cam_in) if using_tensorrt_camencode else model.camencode(cam_in)
                if should_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    stage_t2 = time.perf_counter()
                cam_feats = cam_feats.view(
                    B,
                    N,
                    model.camC,
                    model.D,
                    imH // model.downsample,
                    imW // model.downsample,
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

        # Visualization
        t_vis0 = time.perf_counter()
        cam_tiled = tile_cameras(imgs_cpu)
        bev_img = (bev_prob * 255).astype(np.uint8)
        bev_color = cv2.applyColorMap(bev_img, cv2.COLORMAP_JET)
        cam_tiled = maybe_resize(cam_tiled)
        bev_color = maybe_resize(bev_color)
        draw_ego_triangle(bev_color, grid_conf)
        t_vis_ms = (time.perf_counter() - t_vis0) * 1000.0

        loop_ms_so_far = (time.perf_counter() - loop_t0) * 1000.0
        inst_fps = 1000.0 / max(loop_ms_so_far + fetch_ms, 1e-6)
        fps_ema = inst_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst_fps)
        if using_tensorrt_camencode and using_tensorrt_bevencode:
            backend_label = "TRT(cam+bev)"
        elif using_tensorrt_camencode:
            backend_label = "TRT(cam)"
        elif using_tensorrt_bevencode:
            backend_label = "TRT(bev)"
        else:
            backend_label = "Torch"

        now_t = time.perf_counter()
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
            if eval_tracker is not None:
                eval_summary = eval_tracker.summary()
                with eval_lock:
                    global_eval_stats = {
                        "ready": True,
                        "backend": backend_label,
                        "timestamp_unix": time.time(),
                        **eval_summary,
                    }
                print(
                    f"[EVAL] frames={eval_summary['frames']} "
                    f"IoU={eval_summary['iou']:.3f} P={eval_summary['precision']:.3f} "
                    f"R={eval_summary['recall']:.3f} F1={eval_summary['f1']:.3f} "
                    f"pos_diff={eval_summary['pos_rate_diff']:+.4f} "
                    f"(last_iou={eval_summary['last_iou']:.3f})"
                )
                bins = eval_summary["distance_bins"]
                bin_iou_text = " ".join(
                    f"{label}={vals['iou']:.3f}" for label, vals in bins.items()
                )
                print(
                    "[EVAL-BINS] "
                    f"{bin_iou_text} "
                    f"| near_w_iou={eval_summary['nearfield_weighting']['weighted_iou']:.3f}"
                )
            last_log_t = now_t

        if SHOW_OVERLAY:
            overlay_lines = [
                f"FPS(avg): {fps_ema:.1f} | FPS(inst): {inst_fps:.1f} | Backend: {backend_label}",
                f"Fetch(avg): {fetch_ms_ema:.1f} ms | H2D: {t_h2d_ms:.1f} ms",
                f"Infer: {t_infer_ms:.1f} ms | Copy(avg): {copy_ms_ema:.1f} ms | Vis: {t_vis_ms:.1f} ms | JPEG(avg): {enc_ms_ema:.1f} ms",
            ]
            draw_overlay(cam_tiled, overlay_lines)

        t_enc0 = time.perf_counter()
        cam_ok, cam_jpeg = cv2.imencode(
            '.jpg',
            cam_tiled,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        bev_ok, bev_jpeg = cv2.imencode(
            '.jpg',
            bev_color,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        t_enc_ms = (time.perf_counter() - t_enc0) * 1000.0
        enc_ms_ema = t_enc_ms if enc_ms_ema == 0.0 else (0.9 * enc_ms_ema + 0.1 * t_enc_ms)
        if not (cam_ok and bev_ok):
            continue

        # Update Globals for Server with pre-encoded bytes.
        with frame_lock:
            global_cam_jpeg = cam_jpeg.tobytes()
            global_bev_jpeg = bev_jpeg.tobytes()

        loop_dt = time.perf_counter() - loop_t0
        if loop_dt < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - loop_dt)
            

# --- MAIN ---
if __name__ == "__main__":
    # 1. Start Inference in Background Thread
    t = threading.Thread(target=inference_loop)
    t.daemon = True
    t.start()

    # 2. Start HTTP Server on Main Thread
    print(f"Starting MJPEG Server at http://{HOST_IP}:{PORT}...")
    print(f"  - Camera Stream: http://<JETSON_IP>:{PORT}/cam")
    print(f"  - BEV Stream:    http://<JETSON_IP>:{PORT}/bev")
    
    server = ThreadedHTTPServer((HOST_IP, PORT), MJPEGHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass