"""Image preparation and overlay drawing helpers."""

from typing import Dict, Sequence

import cv2
import numpy as np
import torch

from .config import EGO_LENGTH_M, EGO_WIDTH_M, EGO_X_OFFSET_M, STREAM_SCALE


def denormalize_img(img: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = img.transpose(1, 2, 0)
    img = (img * std + mean) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def tile_cameras(imgs_tensor: torch.Tensor) -> np.ndarray:
    imgs_np = [denormalize_img(img.cpu().numpy()) for img in imgs_tensor]
    top_row = np.hstack(imgs_np[:3])
    bot_row = np.hstack(imgs_np[3:])
    full_grid = np.vstack((top_row, bot_row))
    return cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR)


def draw_overlay(frame: np.ndarray, lines: Sequence[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        y += 22


def maybe_resize(frame: np.ndarray) -> np.ndarray:
    if STREAM_SCALE >= 0.999:
        return frame
    h, w = frame.shape[:2]
    nw = max(1, int(w * STREAM_SCALE))
    nh = max(1, int(h * STREAM_SCALE))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def draw_ego_triangle(frame: np.ndarray, grid_conf: Dict[str, Sequence[float]]) -> None:
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

    row_scale = h / float(nx)
    col_scale = w / float(ny)
    tri_len = max(6, int(round((EGO_LENGTH_M / x_step) * row_scale)))
    tri_half_w = max(3, int(round((EGO_WIDTH_M / y_step) * col_scale * 0.5)))

    tip_y = min(h - 1, cy + tri_len // 2)
    base_y = max(0, cy - tri_len // 2)
    pts = np.array([[cx, tip_y], [cx - tri_half_w, base_y], [cx + tri_half_w, base_y]], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (255, 255, 255))
    cv2.polylines(frame, [pts], True, (0, 0, 0), 2, cv2.LINE_AA)
