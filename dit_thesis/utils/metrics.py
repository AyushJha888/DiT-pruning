"""
utils/metrics.py
----------------
Evaluation metrics: FID, GFLOPs, latency, CSV logging.

All functions are designed to be called from notebooks via:
    from utils import metrics
"""
import os
import time
import csv
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------
def compute_fid(real_dir: str, generated_dir: str) -> float:
    """Compute FID score between real and generated image directories.

    Tries pytorch-fid first; falls back to torchmetrics
    FrechetInceptionDistance if pytorch-fid is not installed.

    Args:
        real_dir (str): Path to directory of real ImageNet validation images.
        generated_dir (str): Path to directory of generated PNG images.

    Returns:
        float: FID score (lower is better).

    Raises:
        ImportError: If neither pytorch-fid nor torchmetrics is installed.
        FileNotFoundError: If either directory does not exist.

    Example:
        fid = compute_fid("data/imagenet_val/", "results/samples/ddim20/")
    """
    for d in (real_dir, generated_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: '{d}'")

    n_real = len([f for f in os.listdir(real_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    n_gen  = len([f for f in os.listdir(generated_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"Computing FID (real: {n_real} images, generated: {n_gen} images)...")

    # --- Try pytorch-fid ---------------------------------------------------------
    try:
        from pytorch_fid import fid_score
        fid = fid_score.calculate_fid_given_paths(
            [real_dir, generated_dir],
            batch_size=64,
            device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            dims=2048,
        )
        print(f"FID: {fid:.2f}")
        return float(fid)
    except ImportError:
        pass

    # --- Fallback: torchmetrics --------------------------------------------------
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader
        from PIL import Image

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fid_metric = FrechetInceptionDistance(feature=2048).to(device)

        tf = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).byte()),
        ])

        def _load_dir(path: str):
            imgs = []
            for fn in os.listdir(path):
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    img = Image.open(os.path.join(path, fn)).convert("RGB")
                    imgs.append(tf(img))
            return imgs

        real_imgs = _load_dir(real_dir)
        gen_imgs  = _load_dir(generated_dir)

        batch_size = 64
        for i in range(0, len(real_imgs), batch_size):
            batch = torch.stack(real_imgs[i:i+batch_size]).to(device)
            fid_metric.update(batch, real=True)
        for i in range(0, len(gen_imgs), batch_size):
            batch = torch.stack(gen_imgs[i:i+batch_size]).to(device)
            fid_metric.update(batch, real=False)

        fid = float(fid_metric.compute())
        print(f"FID: {fid:.2f}")
        return fid

    except ImportError:
        raise ImportError(
            "Neither 'pytorch-fid' nor 'torchmetrics' is installed.\n"
            "Install with:  pip install pytorch-fid\n"
            "           or: pip install torchmetrics[image]"
        )


# ---------------------------------------------------------------------------
# GFLOPs
# ---------------------------------------------------------------------------
def count_flops(model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
    """Count GFLOPs for one forward pass using fvcore.

    The DiT-XL/2 reference value is ~118.64 GFLOPs per forward pass
    on a (1, 4, 32, 32) input latent.

    Args:
        model: DiT model (or patched variant) in eval mode.
        input_tensor: Example input tensor, shape (1, 4, 32, 32).

    Returns:
        float: GFLOPs (giga floating-point operations).

    Raises:
        ImportError: If fvcore is not installed.

    Example:
        gflops = count_flops(model, torch.randn(1, 4, 32, 32).cuda())
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        raise ImportError(
            "fvcore is not installed.\n"
            "Install with: pip install fvcore"
        )

    EXPECTED_GFLOPS = 118.64
    TOLERANCE       = 0.05   # 5%

    model.eval()
    device = input_tensor.device

    # DiT requires timestep + class-label inputs
    t  = torch.zeros(input_tensor.shape[0], dtype=torch.long, device=device)
    y  = torch.zeros(input_tensor.shape[0], dtype=torch.long, device=device)

    with torch.no_grad():
        flops = FlopCountAnalysis(model, (input_tensor, t, y))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        gflops = flops.total() / 1e9

    print(f"GFLOPs: {gflops:.2f}")
    if abs(gflops - EXPECTED_GFLOPS) / EXPECTED_GFLOPS > TOLERANCE:
        print(
            f"WARNING: GFLOPs mismatch — expected ~{EXPECTED_GFLOPS}, "
            f"got {gflops:.2f}"
        )
    return float(gflops)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def measure_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    n_runs: int = 50,
) -> Tuple[float, float]:
    """Measure mean and std wall-clock latency over n_runs forward passes.

    Warms up 10 runs before timing.  Uses CUDA events for GPU timing
    when CUDA is available; falls back to time.perf_counter on CPU.

    Args:
        model: DiT model in eval mode.
        input_tensor: Example input tensor matching the model's expected shape.
        n_runs (int): Number of timed forward passes. Default 50.

    Returns:
        tuple: (mean_ms, std_ms) — mean and standard deviation in milliseconds.

    Example:
        mean_ms, std_ms = measure_latency(model, x)
    """
    device  = input_tensor.device
    use_gpu = device.type == "cuda"

    t  = torch.zeros(input_tensor.shape[0], dtype=torch.long, device=device)
    y  = torch.zeros(input_tensor.shape[0], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(input_tensor, t, y)
        if use_gpu:
            torch.cuda.synchronize()

        # Timed runs
        timings: list = []
        for _ in range(n_runs):
            if use_gpu:
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(input_tensor, t, y)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end))   # ms
            else:
                t0 = time.perf_counter()
                _ = model(input_tensor, t, y)
                timings.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(timings))
    std_ms  = float(np.std(timings))
    print(f"Latency ({n_runs} runs): {mean_ms:.1f}ms ± {std_ms:.1f}ms")
    return mean_ms, std_ms


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------
def log_results(metrics_dict: Dict, csv_path: str) -> None:
    """Append a metrics row (with timestamp) to a CSV file.

    Creates the file and header row if it does not already exist.
    After writing, prints the last 3 rows for a quick sanity check.

    Args:
        metrics_dict (dict): Metric key-value pairs, e.g.
            {"model": "DiT-XL/2", "fid": 2.31, "gflops": 118.64}.
        csv_path (str): Path to the CSV file.

    Returns:
        None

    Example:
        log_results({"model": "DiT-XL/2", "fid": 2.31},
                    "results/baseline_metrics.csv")
    """
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    row = {"timestamp": datetime.datetime.now().isoformat(), **metrics_dict}
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended row to {csv_path}")

    # Print last 3 rows
    with open(csv_path, "r") as f:
        lines = f.readlines()
    n_show = min(3, len(lines))
    print("  Last 3 rows:")
    for line in lines[-n_show:]:
        print("   ", line.rstrip())
