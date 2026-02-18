#!/usr/bin/env python3
"""Profile GPU usage of a launched command, then write CSV + plot.

Example:
  /mnt/nvme/conda_envs/lss-jp512/bin/python scripts/profile_gpu_usage.py \
    --cmd "PYTHONPATH=/mnt/nvme/lss-bev-portfolio /mnt/nvme/conda_envs/lss-jp512/bin/python scripts/benchmark_backends.py --modes trt --num-batches 12 --warmup 2 --measure 8 --output benchmark_results_gpu_profiled.json" \
    --csv assets/gpu_usage_trt.csv \
    --plot assets/gpu_usage_trt.png
"""

import argparse
import csv
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt


def _find_jetson_gpu_load_file() -> Optional[Path]:
    candidates = [
        Path("/sys/devices/gpu.0/load"),
        Path("/sys/devices/17000000.ga10b/load"),
        Path("/sys/devices/platform/17000000.ga10b/load"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _read_jetson_gpu_load_pct(path: Path) -> Optional[float]:
    try:
        raw = path.read_text().strip()
        v = float(raw)
        # Jetson commonly reports 0..1000 where 1000 == 100%
        if v > 100.0:
            return max(0.0, min(100.0, v / 10.0))
        return max(0.0, min(100.0, v))
    except Exception:
        return None


def _nvidia_smi_gpu_util_pct() -> Optional[float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        first = out.splitlines()[0].strip()
        return float(first)
    except Exception:
        return None


def _nvidia_smi_proc_sm_pct(pid: int) -> Optional[float]:
    # pmon columns: gpu pid type sm mem enc dec command
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
    except Exception:
        return None

    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 4:
            continue
        try:
            row_pid = int(parts[1])
        except Exception:
            continue
        if row_pid != pid:
            continue
        sm = parts[3]
        if sm == "-":
            return 0.0
        try:
            return float(sm)
        except Exception:
            return None
    return 0.0


def _proc_stats(pid: int) -> Tuple[Optional[float], Optional[float]]:
    # Returns (cpu_pct, rss_mb). Linux /proc parsing, no psutil dependency.
    try:
        with open(f"/proc/{pid}/statm", "r", encoding="utf-8") as f:
            parts = f.read().strip().split()
            rss_pages = int(parts[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        rss_mb = (rss_pages * page_size) / (1024.0 * 1024.0)
    except Exception:
        rss_mb = None

    # CPU% left as None to keep implementation simple/portable.
    return None, rss_mb


def profile_command(cmd: str, interval_s: float) -> Tuple[int, list, str]:
    nvidia_smi_ok = shutil.which("nvidia-smi") is not None
    jetson_load_file = _find_jetson_gpu_load_file()
    if nvidia_smi_ok:
        backend = "nvidia-smi"
    elif jetson_load_file is not None:
        backend = f"jetson-sysfs:{jetson_load_file}"
    else:
        backend = "unknown"

    proc = subprocess.Popen(cmd, shell=True)
    samples = []
    t0 = time.time()

    try:
        while True:
            ret = proc.poll()
            now = time.time()
            elapsed = now - t0

            gpu_total = None
            gpu_proc = None
            if nvidia_smi_ok:
                gpu_total = _nvidia_smi_gpu_util_pct()
                gpu_proc = _nvidia_smi_proc_sm_pct(proc.pid)
            elif jetson_load_file is not None:
                gpu_total = _read_jetson_gpu_load_pct(jetson_load_file)
                gpu_proc = gpu_total  # best-effort proxy on Jetson sysfs

            cpu_pct, rss_mb = _proc_stats(proc.pid)

            samples.append(
                {
                    "timestamp_s": elapsed,
                    "pid": proc.pid,
                    "gpu_total_util_pct": gpu_total,
                    "gpu_proc_util_pct": gpu_proc,
                    "proc_cpu_pct": cpu_pct,
                    "proc_rss_mb": rss_mb,
                    "backend": backend,
                }
            )

            if ret is not None:
                break
            time.sleep(interval_s)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    return proc.returncode, samples, backend


def write_csv(path: Path, samples: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp_s",
        "pid",
        "gpu_total_util_pct",
        "gpu_proc_util_pct",
        "proc_cpu_pct",
        "proc_rss_mb",
        "backend",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in samples:
            w.writerow(row)


def write_plot(path: Path, samples: list, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [row["timestamp_s"] for row in samples]
    gpu_total = [row["gpu_total_util_pct"] if row["gpu_total_util_pct"] is not None else float("nan") for row in samples]
    gpu_proc = [row["gpu_proc_util_pct"] if row["gpu_proc_util_pct"] is not None else float("nan") for row in samples]
    rss = [row["proc_rss_mb"] if row["proc_rss_mb"] is not None else float("nan") for row in samples]

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(xs, gpu_total, label="GPU util (total)", linewidth=2)
    ax1.plot(xs, gpu_proc, label="GPU util (process/proxy)", linewidth=2, linestyle="--")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("GPU Utilization (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(xs, rss, label="Process RSS (MB)", color="tab:green", alpha=0.7)
    ax2.set_ylabel("RSS Memory (MB)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile GPU usage of a launched command")
    parser.add_argument("--cmd", required=True, help="Command to launch and monitor")
    parser.add_argument("--interval", type=float, default=0.5, help="Sampling interval seconds")
    parser.add_argument("--csv", default="assets/gpu_usage_profile.csv", help="CSV output path")
    parser.add_argument("--plot", default="assets/gpu_usage_profile.png", help="PNG plot output path")
    parser.add_argument("--title", default="GPU Usage Profile", help="Plot title")
    args = parser.parse_args()

    print(f"Launching: {args.cmd}")
    rc, samples, backend = profile_command(args.cmd, args.interval)
    print(f"Sampler backend: {backend}")
    print(f"Target exit code: {rc}")

    csv_path = Path(args.csv)
    plot_path = Path(args.plot)
    write_csv(csv_path, samples)
    write_plot(plot_path, samples, f"{args.title} ({backend})")

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote plot: {plot_path}")

    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
