"""Running BEV metrics + dashboard rendering."""

import time
from html import escape
from typing import Dict

import torch

from .config import (
    EVAL_BIN_MIN_IOU,
    EVAL_DISTANCE_BINS_M,
    EVAL_EGO_X_OFFSET_M,
    EVAL_MIN_F1,
    EVAL_MIN_IOU,
    EVAL_MIN_NEARFIELD_WEIGHTED_IOU,
    EVAL_MIN_PRECISION,
    EVAL_MIN_RECALL,
    EVAL_NEARFIELD_RADIUS_M,
    EVAL_NEARFIELD_WEIGHT,
    GRID_CONF,
)


def _fmt_num(x, nd: int = 3) -> str:
    return f"{float(x):.{nd}f}"


def _status_class(value, threshold) -> str:
    if value is None or threshold is None:
        return "neutral"
    return "pass" if float(value) >= float(threshold) else "fail"


def _metric_cell(name: str, value, threshold=None) -> str:
    cls = _status_class(value, threshold)
    value_txt = _fmt_num(value) if value is not None else "-"
    th_txt = _fmt_num(threshold) if threshold is not None else "-"
    status_txt = "OK" if cls == "pass" else ("LOW" if cls == "fail" else "-")
    return (
        f"<tr class='{cls}'>"
        f"<td>{escape(name)}</td>"
        f"<td>{value_txt}</td>"
        f"<td>{th_txt}</td>"
        f"<td>{status_txt}</td>"
        f"</tr>"
    )


def render_eval_html(metrics: Dict[str, object]) -> str:
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
    updated = time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(float(metrics.get("timestamp_unix", time.time()))),
    )

    rows = []
    for label, vals in bins.items():
        rows.append(_metric_cell(f"{label} IoU", vals.get("iou"), EVAL_BIN_MIN_IOU.get(label)))
    bins_html = "".join(rows) if rows else "<tr class='neutral'><td colspan='4'>No bin stats yet</td></tr>"

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


class RunningBevEval:
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self.distance_bins = tuple((float(lo), float(hi)) for lo, hi in EVAL_DISTANCE_BINS_M)
        self.nearfield_radius_m = float(EVAL_NEARFIELD_RADIUS_M)
        self.nearfield_weight = float(EVAL_NEARFIELD_WEIGHT)
        self._cached_maps = {}
        self.reset()

    def reset(self) -> None:
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
    def _bin_label(lo: float, hi: float) -> str:
        return f"{int(lo)}-{int(hi)}m"

    def _build_distance_maps(self, h: int, w: int, device: torch.device):
        key = (str(device), int(h), int(w))
        if key in self._cached_maps:
            return self._cached_maps[key]

        x_min, x_max = GRID_CONF["xbound"][:2]
        y_min, y_max = GRID_CONF["ybound"][:2]
        x_step = (x_max - x_min) / float(h)
        y_step = (y_max - y_min) / float(w)
        x_centers = x_min + (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * x_step
        y_centers = y_min + (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * y_step
        xx, yy = torch.meshgrid(x_centers, y_centers, indexing="ij")
        dist = torch.sqrt((xx - EVAL_EGO_X_OFFSET_M) ** 2 + (yy - 0.0) ** 2).unsqueeze(0).unsqueeze(0)

        bin_masks = [(dist >= lo) & (dist < hi) for lo, hi in self.distance_bins]
        near_mask = dist <= self.nearfield_radius_m
        weights = torch.ones_like(dist, dtype=torch.float32)
        weights = torch.where(near_mask, torch.full_like(weights, self.nearfield_weight), weights)

        maps = {"bin_masks": bin_masks, "weights": weights}
        self._cached_maps[key] = maps
        return maps

    def update(self, bev_logits: torch.Tensor, bev_target: torch.Tensor) -> None:
        if bev_logits is None or bev_target is None:
            return

        with torch.no_grad():
            pred = torch.sigmoid(bev_logits) >= self.threshold
            tgt = bev_target > 0.5
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

            self.last_iou = tp / max(tp + fp + fn, 1.0)
            self.last_precision = tp / max(tp + fp, 1.0)
            self.last_recall = tp / max(tp + fn, 1.0)
            self.last_f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
            total = max(tp + fp + fn + tn, 1.0)
            self.last_pos_rate_diff = (tp + fp) / total - (tp + fn) / total

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

    def summary(self) -> Dict[str, object]:
        iou = self.tp / max(self.tp + self.fp + self.fn, 1.0)
        precision = self.tp / max(self.tp + self.fp, 1.0)
        recall = self.tp / max(self.tp + self.fn, 1.0)
        f1 = 2.0 * self.tp / max(2.0 * self.tp + self.fp + self.fn, 1.0)
        acc = (self.tp + self.tn) / max(self.tp + self.fp + self.fn + self.tn, 1.0)

        total = max(self.tp + self.fp + self.fn + self.tn, 1.0)
        pred_pos_rate = (self.tp + self.fp) / total
        tgt_pos_rate = (self.tp + self.fn) / total

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
