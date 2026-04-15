"""
utils/wandb_utils.py
--------------------
All wandb logging for the DiT token-pruning thesis.

RULE: Never call wandb.log() directly in notebooks — always use these wrappers.

Usage:
    from utils import wandb_utils
    run = wandb_utils.init_run("baseline-ddpm250", tags=["baseline"])
    wandb_utils.log_metrics({"fid": 2.31}, step=0)
    wandb_utils.finish_run()
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import wandb

# Module-level handle so all functions share the same active run.
_current_run: Optional[wandb.sdk.wandb_run.Run] = None


def _get_run() -> wandb.sdk.wandb_run.Run:
    """Return the active run or raise if none has been initialised."""
    if _current_run is None or _current_run._is_finished:
        raise RuntimeError(
            "No active wandb run. Call wandb_utils.init_run() first."
        )
    return _current_run


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------
def init_run(
    run_name: str,
    tags: Optional[List[str]] = None,
    config_override: Optional[Dict[str, Any]] = None,
) -> wandb.sdk.wandb_run.Run:
    """Initialise a wandb run.  Call once at the top of each notebook.

    Merges config.WANDB_TAGS with any extra tags provided.
    Builds a full run config from utils/config.py, then applies overrides.

    Args:
        run_name (str): Short identifier, e.g. "baseline-ddim20".
        tags (list[str]): Additional tags appended to config.WANDB_TAGS.
        config_override (dict): Key-value pairs to override in the run config.

    Returns:
        wandb.Run: The initialised run object.

    Example:
        run = init_run("baseline-ddpm250", tags=["ddpm"],
                       config_override={"steps": 250})
    """
    global _current_run

    # Lazy import config to avoid circular dependency
    from utils import config as cfg

    tags = tags or []
    config_override = config_override or {}

    all_tags = list(cfg.WANDB_TAGS) + tags

    run_config = {
        "model_path":        cfg.MODEL_PATH,
        "image_size":        cfg.IMAGE_SIZE,
        "num_classes":       cfg.NUM_CLASSES,
        "ddpm_steps":        cfg.DDPM_STEPS,
        "ddim_steps":        cfg.DDIM_STEPS,
        "num_samples_quick": cfg.NUM_SAMPLES_QUICK,
        "num_samples_fid":   cfg.NUM_SAMPLES_FID,
        "device":            cfg.DEVICE,
        "seed":              cfg.SEED,
        **config_override,
    }

    _current_run = wandb.init(
        project=cfg.WANDB_PROJECT,
        entity=cfg.WANDB_ENTITY,
        name=run_name,
        tags=all_tags,
        config=run_config,
        reinit=True,
    )

    print(f"wandb run initialized: {_current_run.get_url()}")
    return _current_run


def finish_run() -> None:
    """Mark the current wandb run as complete and print the final URL.

    Example:
        wandb_utils.finish_run()
    """
    global _current_run
    run = _get_run()
    url = run.get_url()
    wandb.finish()
    _current_run = None
    print(f"Run complete. View at: {url}")


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def log_metrics(
    metrics_dict: Dict[str, float],
    step: Optional[int] = None,
) -> None:
    """Log scalar metrics to the active wandb run.

    Args:
        metrics_dict (dict): e.g. {"fid": 2.31, "gflops": 118.64}.
        step (int): Optional global step counter.

    Example:
        log_metrics({"fid": 2.31, "gflops": 118.64}, step=0)
    """
    run = _get_run()
    payload = dict(metrics_dict)
    if step is not None:
        payload["_step"] = step
        run.log(payload, step=step)
    else:
        run.log(payload)


def log_images(
    images,
    caption: str,
    step: Optional[int] = None,
    max_images: int = 8,
) -> None:
    """Log PIL images to the active wandb run (capped at max_images).

    Args:
        images (list[PIL.Image]): Images to log.
        caption (str): e.g. "DDPM-250 samples".
        step (int): Optional global step.
        max_images (int): Maximum number of images to log. Default 8.

    Example:
        log_images(pil_list, caption="DDIM-20 samples", step=0)
    """
    run = _get_run()
    imgs_to_log = images[:max_images]
    wandb_imgs  = [wandb.Image(img, caption=f"{caption} [{i}]")
                   for i, img in enumerate(imgs_to_log)]
    payload = {caption: wandb_imgs}
    if step is not None:
        run.log(payload, step=step)
    else:
        run.log(payload)
    print(f"Logged {len(imgs_to_log)} sample images to wandb")


def log_figure(fig, name: str, step: Optional[int] = None) -> None:
    """Log a matplotlib Figure to the active wandb run.

    Args:
        fig (matplotlib.Figure): Figure to log.
        name (str): Key name, e.g. "token_importance_t1".
        step (int): Optional global step.

    Example:
        log_figure(fig, name="token_importance_t1", step=1)
    """
    run = _get_run()
    payload = {name: wandb.Image(fig)}
    if step is not None:
        run.log(payload, step=step)
    else:
        run.log(payload)


def log_table(df, table_name: str) -> None:
    """Log a pandas DataFrame as a wandb Table.

    Args:
        df (pd.DataFrame): DataFrame to log.
        table_name (str): e.g. "baseline_comparison".

    Example:
        log_table(results_df, "baseline_comparison")
    """
    run = _get_run()
    table = wandb.Table(dataframe=df)
    run.log({table_name: table})


def log_artifact(path: str, name: str, artifact_type: str) -> None:
    """Log a file or directory as a versioned wandb artifact.

    Args:
        path (str): Local file or directory path.
        name (str): Artifact name, e.g. "saliency-masks".
        artifact_type (str): "dataset" or "model".

    Raises:
        FileNotFoundError: If path does not exist.

    Example:
        log_artifact("results/saliency_maps/", "saliency-masks", "dataset")
        log_artifact("results/tdw_router.pt",  "tdw-router-v1",  "model")
    """
    run = _get_run()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact path not found: '{path}'")

    artifact = wandb.Artifact(name=name, type=artifact_type)
    if os.path.isdir(path):
        artifact.add_dir(path)
    else:
        artifact.add_file(path)

    run.log_artifact(artifact)
    print(f"Artifact '{name}' logged to wandb")
