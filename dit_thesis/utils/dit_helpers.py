"""
utils/dit_helpers.py
--------------------
Load DiT-XL/2 + VAE, run sampling (DDPM / DDIM), save images.

All public functions are designed to be called from notebooks via:
    from utils import dit_helpers
"""
import os
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Optional imports — gracefully degrade if DiT repo is not on sys.path
# ---------------------------------------------------------------------------
try:
    from models import DiT_XL_2                        # official DiT repo
    from diffusion import create_diffusion              # official DiT repo
    from diffusers.models import AutoencoderKL
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Seed helper (also exported for notebooks)
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed (int): Seed value. Default 42.

    Example:
        seed_everything(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(config) -> Tuple:
    """Load DiT-XL/2 and SD VAE from paths specified in config.

    Args:
        config: The config module (utils/config.py).

    Returns:
        tuple: (dit_model, vae, diffusion)
            - dit_model : DiT-XL/2 in eval mode on config.DEVICE
            - vae       : AutoencoderKL in eval mode on config.DEVICE
            - diffusion : GaussianDiffusion schedule object

    Raises:
        ImportError: If the DiT repository or diffusers are not installed.
        FileNotFoundError: If pretrained weight files are missing.

    Example:
        model, vae, diffusion = load_model(config)
    """
    if not _DEPS_AVAILABLE:
        raise ImportError(
            "DiT repo or diffusers not found on sys.path.\n"
            "Add the official DiT repository to sys.path before calling load_model().\n"
            "  import sys; sys.path.insert(0, '/path/to/DiT')"
        )

    device = config.DEVICE

    # --- DiT-XL/2 ----------------------------------------------------------------
    print(f"Loading DiT-XL/2 from {config.MODEL_PATH}...")
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"DiT weights not found at '{config.MODEL_PATH}'.\n"
            "Download from: https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt"
        )

    dit_model = DiT_XL_2(input_size=config.LATENT_SIZE).to(device)
    state_dict = torch.load(config.MODEL_PATH, map_location=device)
    dit_model.load_state_dict(state_dict)
    dit_model.eval()

    n_params_dit = sum(p.numel() for p in dit_model.parameters()) / 1e6
    print(f"DiT-XL/2 loaded: {n_params_dit:.0f}M params")

    # --- VAE ----------------------------------------------------------------------
    print(f"Loading VAE from {config.VAE_PATH}...")
    vae = AutoencoderKL.from_pretrained(config.VAE_PATH).to(device)
    vae.eval()

    n_params_vae = sum(p.numel() for p in vae.parameters()) / 1e6
    print(f"VAE loaded: {n_params_vae:.0f}M params")

    # --- Diffusion schedule -------------------------------------------------------
    diffusion = create_diffusion(timestep_respacing="")   # full 1000-step schedule
    print(f"Diffusion schedule: DDPM, 1000 steps")

    return dit_model, vae, diffusion


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def run_sampling(
    model,
    vae,
    diffusion,
    n_samples: int,
    steps: int,
    sampler: str = "ddpm",
    seed: int = 42,
    cfg_scale: float = 4.0,
    num_classes: int = 1000,
    latent_size: int = 32,
    device: Optional[str] = None,
) -> List[Image.Image]:
    """Run denoising sampling and decode to a list of PIL images.

    Supports DDPM (slow, high quality) and DDIM (fast) samplers.
    Uses classifier-free guidance (CFG) matching the original DiT paper.

    Args:
        model: DiT-XL/2 model in eval mode.
        vae: AutoencoderKL decoder in eval mode.
        diffusion: GaussianDiffusion schedule object.
        n_samples (int): Number of images to generate.
        steps (int): Number of denoising steps.
        sampler (str): "ddpm" or "ddim". Default "ddpm".
        seed (int): Random seed for reproducibility. Default 42.
        cfg_scale (float): Classifier-free guidance scale. Default 4.0.
        num_classes (int): Number of ImageNet classes. Default 1000.
        latent_size (int): Spatial size of latent (IMAGE_SIZE // 8). Default 32.
        device (str): Target device. Auto-detected if None.

    Returns:
        list[PIL.Image]: Decoded RGB images, each 256×256.

    Raises:
        ValueError: If sampler is not "ddpm" or "ddim".

    Example:
        images = run_sampling(model, vae, diffusion, n_samples=50,
                              steps=20, sampler="ddim")
    """
    if sampler not in ("ddpm", "ddim"):
        raise ValueError(f"sampler must be 'ddpm' or 'ddim', got '{sampler}'")

    if device is None:
        device = next(model.parameters()).device

    seed_everything(seed)

    sampler_tag = f"{sampler.upper()}-{steps}"
    print(f"Sampling {n_samples} images with {sampler_tag}...")

    # Build timestep respacing string for diffusion library
    if sampler == "ddim":
        respacing = f"ddim{steps}"
    else:
        respacing = str(steps)

    from diffusion import create_diffusion as _create_diffusion
    sample_diffusion = _create_diffusion(timestep_respacing=respacing)
    sample_fn = (
        sample_diffusion.ddim_sample_loop
        if sampler == "ddim"
        else sample_diffusion.p_sample_loop
    )

    images: List[Image.Image] = []
    t0 = time.time()

    with torch.no_grad():
        # Sample random ImageNet class labels
        class_labels = torch.randint(0, num_classes, (n_samples,), device=device)
        # Duplicate for CFG (unconditional branch uses class = num_classes)
        y_null = torch.full_like(class_labels, num_classes)
        y_combined = torch.cat([class_labels, y_null], dim=0)

        for i in tqdm(range(0, n_samples, 1), total=n_samples, desc=sampler_tag):
            z = torch.randn(2, 4, latent_size, latent_size, device=device)
            model_kwargs = dict(
                y=y_combined[i : i + 2] if n_samples > 1 else y_combined,
                cfg_scale=cfg_scale,
            )
            # Generate single latent
            z_i = torch.randn(1, 4, latent_size, latent_size, device=device)
            z_i = torch.cat([z_i, z_i], dim=0)
            y_i = torch.tensor([class_labels[i], num_classes], device=device)
            model_kwargs_i = dict(y=y_i, cfg_scale=cfg_scale)

            sample = sample_fn(
                model.forward_with_cfg,
                z_i.shape,
                z_i,
                clip_denoised=False,
                model_kwargs=model_kwargs_i,
                progress=False,
                device=device,
            )
            # Keep only the CFG-guided half
            sample = sample[:1]

            # Decode latent → pixel space
            sample = vae.decode(sample / 0.18215).sample
            sample = (sample.clamp(-1, 1) + 1) / 2   # [0, 1]
            sample = (sample * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
            images.append(Image.fromarray(sample[0]))

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s — {n_samples} images generated")
    return images


# ---------------------------------------------------------------------------
# Save samples
# ---------------------------------------------------------------------------
def save_samples(images: List[Image.Image], out_dir: str) -> List[str]:
    """Save a list of PIL images as numbered PNG files.

    Args:
        images (list[PIL.Image]): Images to save.
        out_dir (str): Output directory. Created if it does not exist.

    Returns:
        list[str]: Absolute paths of saved files.

    Example:
        paths = save_samples(images, "results/samples/ddim20/")
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        p = os.path.join(out_dir, f"sample_{i:04d}.png")
        img.save(p)
        paths.append(p)
    print(f"Saved {len(images)} images to {out_dir}")
    return paths
