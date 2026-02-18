#!/usr/bin/env python3
"""Capture GIFs from MJPEG HTTP endpoints.

Example:
  python scripts/capture_mjpeg_gif.py \
    --cam-url http://localhost:8080/cam \
    --bev-url http://localhost:8080/bev \
    --duration 6 --fps 8 --out-dir .
"""

import argparse
import io
import time
import urllib.request
from pathlib import Path

from PIL import Image


def _read_next_jpeg(stream, buffer: bytearray):
    while True:
        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9")
        if start != -1 and end != -1 and end > start:
            jpeg = bytes(buffer[start : end + 2])
            del buffer[: end + 2]
            return jpeg
        chunk = stream.read(4096)
        if not chunk:
            return None
        buffer.extend(chunk)


def capture_gif(url: str, output_path: Path, duration_s: float, fps: float, resize_width: int = 0):
    frame_interval = 1.0 / max(fps, 1e-6)
    frame_count_target = max(1, int(round(duration_s * fps)))

    req = urllib.request.Request(url, headers={"User-Agent": "mjpeg-gif-capture/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        buffer = bytearray()
        frames = []
        next_frame_t = time.perf_counter()

        while len(frames) < frame_count_target:
            jpeg = _read_next_jpeg(resp, buffer)
            if jpeg is None:
                break

            now = time.perf_counter()
            if now < next_frame_t:
                continue
            next_frame_t += frame_interval

            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            if resize_width and resize_width > 0:
                w, h = img.size
                new_h = max(1, int(round(h * (resize_width / float(w)))))
                img = img.resize((resize_width, new_h), Image.Resampling.LANCZOS)
            frames.append(img)

    if not frames:
        raise RuntimeError(f"No frames captured from {url}")

    duration_ms = int(round(1000.0 / max(fps, 1e-6)))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return len(frames)


def main():
    parser = argparse.ArgumentParser(description="Capture GIFs from MJPEG endpoints")
    parser.add_argument("--cam-url", default="http://localhost:8080/cam")
    parser.add_argument("--bev-url", default="http://localhost:8080/bev")
    parser.add_argument("--duration", type=float, default=6.0, help="Duration in seconds")
    parser.add_argument("--fps", type=float, default=8.0, help="GIF frame rate")
    parser.add_argument("--out-dir", default=".", help="Directory to write GIFs")
    parser.add_argument("--resize-width", type=int, default=0, help="Resize width for GIFs (0 keeps original)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_path = out_dir / "cam.gif"
    bev_path = out_dir / "bev.gif"

    print(f"Capturing CAM GIF from {args.cam_url} ...")
    cam_n = capture_gif(args.cam_url, cam_path, args.duration, args.fps, args.resize_width)
    print(f"Saved {cam_path} ({cam_n} frames)")

    print(f"Capturing BEV GIF from {args.bev_url} ...")
    bev_n = capture_gif(args.bev_url, bev_path, args.duration, args.fps, args.resize_width)
    print(f"Saved {bev_path} ({bev_n} frames)")


if __name__ == "__main__":
    main()
