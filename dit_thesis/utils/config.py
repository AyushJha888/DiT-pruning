"""
utils/config.py
---------------
Central configuration for the DiT token-pruning thesis.
Import this module; it auto-prints a summary table on first load.

Usage:
    from utils import config
    model_path = config.MODEL_PATH
"""
import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = "pretrained_models/DiT-XL-2-256x256.pt"
VAE_PATH   = "pretrained_models/sd-vae-ft-ema"

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Model / image settings
# ---------------------------------------------------------------------------
IMAGE_SIZE   = 256
NUM_CLASSES  = 1000
LATENT_SIZE  = IMAGE_SIZE // 8   # 32  — VAE downsamples 8×
IN_CHANNELS  = 4                 # latent channels
PATCH_SIZE   = 2                 # DiT-XL/2 patch size
NUM_TOKENS   = (LATENT_SIZE // PATCH_SIZE) ** 2  # 256

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
DDPM_STEPS        = 250
DDIM_STEPS        = 20
CFG_SCALE         = 4.0          # classifier-free guidance scale
NUM_SAMPLES_QUICK = 50           # for visual checks
NUM_SAMPLES_FID   = 10_000       # for real FID

# ---------------------------------------------------------------------------
# Results & logging
# ---------------------------------------------------------------------------
RESULTS_DIR  = "results/"
SALIENCY_DIR = "results/saliency_maps/"

WANDB_PROJECT = "dit-token-pruning-thesis"
WANDB_ENTITY  = "rahul23082001jha"
WANDB_TAGS    = ["phase1"]

# ---------------------------------------------------------------------------
# Pretty-print summary table on import
# ---------------------------------------------------------------------------
def _print_config_table() -> None:
    rows = [
        ("Device",           DEVICE),
        ("Image size",       str(IMAGE_SIZE)),
        ("DDPM steps",       str(DDPM_STEPS)),
        ("DDIM steps",       str(DDIM_STEPS)),
        ("Samples (quick)",  str(NUM_SAMPLES_QUICK)),
        ("Samples (FID)",    str(NUM_SAMPLES_FID)),
        ("wandb project",    WANDB_PROJECT[:14] + "..."),
    ]
    key_w, val_w = 21, 16
    border_top    = "┌" + "─" * (key_w + 2) + "┬" + "─" * (val_w + 2) + "┐"
    header_line   = "│" + " DiT Thesis — Config".center(key_w + val_w + 5) + "│"
    border_mid    = "├" + "─" * (key_w + 2) + "┼" + "─" * (val_w + 2) + "┤"
    border_bot    = "└" + "─" * (key_w + 2) + "┴" + "─" * (val_w + 2) + "┘"

    print(border_top)
    print(header_line)
    print(border_mid)
    for k, v in rows:
        print(f"│ {k:<{key_w}} │ {v:<{val_w}} │")
    print(border_bot)


_print_config_table()
