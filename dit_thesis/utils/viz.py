"""
utils/viz.py
------------
All visualization helpers for the DiT token-pruning thesis.

Conventions:
  - Every function returns a matplotlib.figure.Figure.
  - Every function accepts save_path=None (skips saving when None).
  - Every function calls plt.show() for inline display in notebooks.
  - Default style: seaborn "whitegrid", figsize=(10, 6).

Usage:
    from utils import viz
    fig = viz.plot_token_grid(scores, timestep=5)
"""
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from PIL import Image

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    plt.style.use("ggplot")

DEFAULT_FIGSIZE = (10, 6)
CMAP_ATTN      = "viridis"
CMAP_IMPORTANCE= "hot"


def _save_and_show(fig: Figure, save_path: Optional[str]) -> Figure:
    """Internal: optionally save figure, then call plt.show()."""
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 1. Attention heatmap
# ---------------------------------------------------------------------------
def plot_heatmap(
    attn_map: np.ndarray,
    timestep: int,
    layer: int,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot a single attention head heatmap (seq_len × seq_len).

    Args:
        attn_map (np.ndarray): Shape (seq_len, seq_len). Values in [0, 1].
        timestep (int): Denoising timestep label for the title.
        layer (int): DiT block index for the title.
        save_path (str): Optional path to save the figure as PNG.

    Returns:
        matplotlib.Figure

    Example:
        fig = plot_heatmap(attn_map, timestep=1, layer=0)
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    im = ax.imshow(attn_map, cmap=CMAP_ATTN, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("attention weight", fontsize=11)
    ax.set_title(f"Attention map — layer {layer}, t={timestep}", fontsize=13)
    ax.set_xlabel("key token index")
    ax.set_ylabel("query token index")
    fig.tight_layout()
    return _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Token importance grid
# ---------------------------------------------------------------------------
def plot_token_grid(
    importance_scores: np.ndarray,
    grid_shape: Tuple[int, int] = (16, 16),
    timestep: Optional[int] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Visualize token importance as a spatial heatmap.

    Args:
        importance_scores (np.ndarray): Shape (256,) — one score per token.
        grid_shape (tuple): (16, 16) for DiT-XL/2 at 256×256.
        timestep (int): Optional timestep label for the title.
        save_path (str): Optional save path.

    Returns:
        matplotlib.Figure

    Example:
        fig = plot_token_grid(scores, timestep=5)
    """
    grid = importance_scores.reshape(grid_shape)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(grid, cmap=CMAP_IMPORTANCE, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("importance score", fontsize=11)

    title = "Token importance"
    if timestep is not None:
        title += f" — t={timestep}"
    ax.set_title(title, fontsize=13)

    # Grid lines at token boundaries
    rows, cols = grid_shape
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.3, alpha=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    return _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Overlay pruning mask on image
# ---------------------------------------------------------------------------
def overlay_mask(
    image: Image.Image,
    binary_mask: np.ndarray,
    alpha: float = 0.5,
    save_path: Optional[str] = None,
) -> Figure:
    """Overlay a binary pruning mask on a PIL image.

    Pruned tokens (mask=0) are tinted red; kept tokens (mask=1) are transparent.

    Args:
        image (PIL.Image): Original generated image (256×256 RGB).
        binary_mask (np.ndarray): Shape (16, 16). 0=prune, 1=keep.
        alpha (float): Overlay opacity for pruned regions. Default 0.5.
        save_path (str): Optional save path.

    Returns:
        matplotlib.Figure

    Example:
        fig = overlay_mask(pil_img, mask_array, alpha=0.5)
    """
    img_arr = np.array(image.convert("RGB"))
    H, W    = img_arr.shape[:2]

    # Upsample (16,16) → (H,W) using nearest-neighbour
    mask_up = np.repeat(np.repeat(binary_mask, H // 16, axis=0),
                        W // 16, axis=1)

    # Red overlay on pruned regions (mask_up == 0)
    overlay      = img_arr.copy().astype(float)
    pruned_mask  = (mask_up == 0)
    overlay[pruned_mask, 0] = 255 * alpha + overlay[pruned_mask, 0] * (1 - alpha)
    overlay[pruned_mask, 1] = overlay[pruned_mask, 1] * (1 - alpha)
    overlay[pruned_mask, 2] = overlay[pruned_mask, 2] * (1 - alpha)
    overlay = overlay.clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_arr)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Pruning mask (red=pruned, α={alpha})", fontsize=12)
    axes[1].axis("off")

    kept_pct   = float(binary_mask.mean()) * 100
    pruned_pct = 100 - kept_pct
    fig.suptitle(
        f"Token mask — {kept_pct:.0f}% kept · {pruned_pct:.0f}% pruned",
        fontsize=13,
    )
    fig.tight_layout()
    return _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Cumulative importance curves
# ---------------------------------------------------------------------------
def plot_cumulative_importance(
    importance_scores_per_timestep: Dict[int, np.ndarray],
    threshold: float = 0.5,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot cumulative importance curve with threshold line marked.

    Shows what fraction of total attention is carried by the top-k% of tokens.

    Args:
        importance_scores_per_timestep (dict): {timestep: np.ndarray shape (256,)}.
        threshold (float): Mark this token fraction on the plot. Default 0.5.
        save_path (str): Optional save path.

    Returns:
        matplotlib.Figure

    Example:
        fig = plot_cumulative_importance({1: scores_t1, 20: scores_t20})
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    colors  = plt.cm.plasma(np.linspace(0.1, 0.9, len(importance_scores_per_timestep)))

    for color, (t, scores) in zip(colors, importance_scores_per_timestep.items()):
        sorted_desc = np.sort(scores)[::-1]
        cumsum      = np.cumsum(sorted_desc)
        cumsum      = cumsum / cumsum[-1]            # normalise to [0, 1]
        fracs       = np.linspace(0, 1, len(cumsum))
        ax.plot(fracs, cumsum, label=f"t={t}", color=color, linewidth=2)

        # Annotate at threshold
        idx     = np.searchsorted(fracs, threshold)
        idx     = min(idx, len(cumsum) - 1)
        pct_val = cumsum[idx] * 100
        ax.annotate(
            f"top {threshold*100:.0f}% → {pct_val:.0f}% attn at t={t}",
            xy=(fracs[idx], cumsum[idx]),
            xytext=(fracs[idx] + 0.05, cumsum[idx] - 0.06),
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        )

    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=1.4,
               label=f"threshold = {threshold}")
    ax.set_xlabel("Fraction of tokens (sorted by importance)", fontsize=12)
    ax.set_ylabel("Cumulative attention", fontsize=12)
    ax.set_title("Cumulative token importance", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 5. TDW width schedule
# ---------------------------------------------------------------------------
def plot_width_schedule(
    width_ratios: Union[List[float], np.ndarray],
    timesteps: Optional[List[int]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot TDW router width ratio vs timestep.

    Args:
        width_ratios (list or np.ndarray): r_t values in [0.25, 1.0].
            If a 2-D array of shape (n_runs, T), a shaded std band is plotted.
        timesteps (list[int]): Optional x-axis labels. Defaults to range(1, T+1).
        save_path (str): Optional save path.

    Returns:
        matplotlib.Figure

    Example:
        fig = plot_width_schedule(r_t_list)
    """
    ratios = np.asarray(width_ratios)
    if ratios.ndim == 1:
        mean_r = ratios
        std_r  = None
    else:
        mean_r = ratios.mean(axis=0)
        std_r  = ratios.std(axis=0)

    T = len(mean_r)
    xs = timesteps if timesteps is not None else list(range(1, T + 1))

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.plot(xs, mean_r, color="#2196F3", linewidth=2, label="width ratio $r_t$")

    if std_r is not None:
        ax.fill_between(xs, mean_r - std_r, mean_r + std_r,
                        alpha=0.25, color="#2196F3", label="±1 std")

    ax.axhline(y=1.00, color="green",  linestyle="--", linewidth=1.2,
               label="full width (1.0)")
    ax.axhline(y=0.25, color="tomato", linestyle="--", linewidth=1.2,
               label="min clamp (0.25)")

    # Annotate minimum
    min_idx = int(np.argmin(mean_r))
    min_t   = xs[min_idx]
    min_val = mean_r[min_idx]
    ax.annotate(
        f"min $r_t$ = {min_val:.2f} at t={min_t}",
        xy=(min_t, min_val),
        xytext=(min_t + T * 0.05, min_val + 0.08),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", lw=1.2),
    )

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Width ratio $r_t$", fontsize=12)
    ax.set_title("TDW Router — width ratio vs timestep", fontsize=13)
    ax.set_ylim(0.15, 1.1)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 6. Comparison table
# ---------------------------------------------------------------------------
def plot_comparison_table(
    df,
    title: str = "Phase 1 Results",
    save_path: Optional[str] = None,
) -> Figure:
    """Render a pandas DataFrame as a styled matplotlib table figure.

    Best value in each numeric column is highlighted green.
    Any FID column value where delta vs. first row > 0.5 is highlighted red.

    Args:
        df (pd.DataFrame): Results dataframe.
        title (str): Figure title. Default "Phase 1 Results".
        save_path (str): Optional save path.

    Returns:
        matplotlib.Figure

    Example:
        fig = plot_comparison_table(results_df, title="Baseline vs TDW")
    """
    import pandas as pd

    fig, ax = plt.subplots(figsize=(max(10, len(df.columns) * 1.8),
                                    max(3, len(df) * 0.7 + 1.5)))
    ax.axis("off")

    col_labels = list(df.columns)
    cell_text  = [
        [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
        for row in df.itertuples(index=False)
    ]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.6)

    # Header style
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#37474F")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Identify numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Highlight best value green; FID delta >0.5 red
    for col in numeric_cols:
        col_j  = col_labels.index(col)
        values = df[col].values

        # Determine best: FID lower is better, others vary
        lower_is_better = any(k in col.lower() for k in ("fid", "latency", "flop", "gflop"))
        best_val        = values.min() if lower_is_better else values.max()

        fid_col    = "fid" in col.lower()
        base_fid   = float(df[col].iloc[0]) if fid_col else None

        for row_i, val in enumerate(values):
            cell = tbl[(row_i + 1, col_j)]
            if val == best_val:
                cell.set_facecolor("#A5D6A7")   # green
            if fid_col and base_fid is not None and abs(float(val) - base_fid) > 0.5:
                cell.set_facecolor("#EF9A9A")   # red

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    return _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 7. Image grid
# ---------------------------------------------------------------------------
def plot_image_grid(
    images: List[Image.Image],
    title: str = "",
    ncols: int = 4,
    save_path: Optional[str] = None,
) -> Figure:
    """Display a list of PIL images in a grid.

    Args:
        images (list[PIL.Image]): Images to display.
        title (str): Figure title.
        ncols (int): Number of columns. Default 4.
        save_path (str): Optional save path.

    Returns:
        matplotlib.Figure

    Example:
        fig = plot_image_grid(images[:8], title="DDPM-250 samples", ncols=4)
    """
    n     = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i])
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 8. Side-by-side image comparison grid
# ---------------------------------------------------------------------------
def plot_side_by_side_grid(
    images_a: List[Image.Image],
    images_b: List[Image.Image],
    labels: List[str],
    title: str = "",
    save_path: Optional[str] = None,
) -> Figure:
    """Display two equal-length image lists side by side for comparison.

    Args:
        images_a (list[PIL.Image]): First set of images (left column group).
        images_b (list[PIL.Image]): Second set of images (right column group).
        labels (list[str]): Two labels, e.g. ["DDPM-250", "DDIM-20"].
        title (str): Figure title.
        save_path (str): Optional save path.

    Returns:
        matplotlib.Figure

    Example:
        fig = plot_side_by_side_grid(ddpm[:4], ddim[:4],
                                     labels=["DDPM-250", "DDIM-20"])
    """
    assert len(images_a) == len(images_b), "Image lists must have equal length."
    n = len(images_a)

    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        axes[i, 0].imshow(images_a[i])
        axes[i, 1].imshow(images_b[i])
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")

    axes[0, 0].set_title(labels[0], fontsize=13, pad=6)
    axes[0, 1].set_title(labels[1], fontsize=13, pad=6)

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 9. Attention map grid (layers × timesteps)
# ---------------------------------------------------------------------------
def plot_attention_grid(
    attn_store: Dict,
    layers: List[int],
    timesteps: List[int],
    title: str = "",
    save_path: Optional[str] = None,
) -> Figure:
    """Plot a grid of attention maps: rows=layers, cols=timesteps.

    Args:
        attn_store (dict): {(layer_idx, timestep): np.ndarray shape (seq, seq)}.
        layers (list[int]): Layer indices to display.
        timesteps (list[int]): Timestep indices to display.
        title (str): Figure title.
        save_path (str): Optional save path.

    Returns:
        matplotlib.Figure

    Example:
        fig = plot_attention_grid(attn_store, layers=[0,9,18,27],
                                  timesteps=[1,5,10,20])
    """
    nrows, ncols = len(layers), len(timesteps)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for ri, layer in enumerate(layers):
        for ci, t in enumerate(timesteps):
            ax   = axes[ri, ci]
            key  = (layer, t)
            if key in attn_store:
                amap = np.asarray(attn_store[key])
                # If multi-head: average over heads
                if amap.ndim == 3:
                    amap = amap.mean(axis=0)
                elif amap.ndim == 4:
                    amap = amap.mean(axis=(0, 1))
                ax.imshow(amap, cmap=CMAP_ATTN, aspect="auto")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"t={t}", fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"Layer {layer}", fontsize=11)

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    return _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 10. Importance violin plot
# ---------------------------------------------------------------------------
def plot_importance_violin(
    importance_scores: np.ndarray,
    timesteps: List[int],
    title: str = "",
    save_path: Optional[str] = None,
) -> Figure:
    """Violin plot of token importance distribution per timestep.

    Args:
        importance_scores (np.ndarray): Shape (T, 256).
        timesteps (list[int]): x-axis labels (length T).
        title (str): Figure title.
        save_path (str): Optional save path.

    Returns:
        matplotlib.Figure

    Example:
        fig = plot_importance_violin(importance_scores,
                                     timesteps=list(range(1, 21)))
    """
    fig, ax = plt.subplots(figsize=(max(10, len(timesteps) * 0.7), 6))

    parts = ax.violinplot(
        [importance_scores[i] for i in range(len(timesteps))],
        positions=list(range(len(timesteps))),
        showmedians=True,
        showextrema=True,
    )

    for pc in parts["bodies"]:
        pc.set_facecolor("#64B5F6")
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(timesteps)))
    ax.set_xticklabels([str(t) for t in timesteps], fontsize=9, rotation=45)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Importance score", fontsize=12)
    ax.set_title(title or "Token importance distribution across timesteps", fontsize=13)
    fig.tight_layout()
    return _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 11. Saliency comparison (DDPM vs DDIM at relative steps)
# ---------------------------------------------------------------------------
def plot_saliency_comparison(
    ddpm_scores: np.ndarray,
    ddim_scores: np.ndarray,
    relative_steps: List[float],
    title: str = "",
    save_path: Optional[str] = None,
) -> Figure:
    """Side-by-side token grids comparing DDPM and DDIM saliency.

    Args:
        ddpm_scores (np.ndarray): Shape (T_ddpm, 256).
        ddim_scores (np.ndarray): Shape (T_ddim, 256).
        relative_steps (list[float]): Fractions of trajectory, e.g. [0.1, 0.3, 0.6, 1.0].
        title (str): Figure title.
        save_path (str): Optional save path.

    Returns:
        matplotlib.Figure

    Example:
        fig = plot_saliency_comparison(ddpm_scores, ddim_scores,
                                       relative_steps=[0.1, 0.3, 0.6, 1.0])
    """
    T_ddpm, T_ddim = ddpm_scores.shape[0], ddim_scores.shape[0]
    n_pairs = len(relative_steps)

    fig, axes = plt.subplots(n_pairs, 2, figsize=(8, n_pairs * 3.5))
    if n_pairs == 1:
        axes = axes[np.newaxis, :]

    for ri, rel in enumerate(relative_steps):
        idx_ddpm = min(int(rel * T_ddpm), T_ddpm - 1)
        idx_ddim = min(int(rel * T_ddim), T_ddim - 1)

        for ci, (scores, label, idx) in enumerate([
            (ddpm_scores, "DDPM-250", idx_ddpm),
            (ddim_scores, "DDIM-20",  idx_ddim),
        ]):
            ax   = axes[ri, ci]
            grid = scores[idx].reshape(16, 16)
            im   = ax.imshow(grid, cmap=CMAP_IMPORTANCE, aspect="equal",
                             vmin=0, vmax=scores.max())
            ax.set_xticks([])
            ax.set_yticks([])
            t_label = idx + 1
            ax.set_title(f"{label}  t={t_label} (rel={rel:.1f})", fontsize=10)

    # Shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.7])
    sm      = plt.cm.ScalarMappable(cmap=CMAP_IMPORTANCE)
    sm.set_clim(0, max(ddpm_scores.max(), ddim_scores.max()))
    fig.colorbar(sm, cax=cbar_ax, label="importance score")

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.87, 1])
    return _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 12. Panel of existing figures
# ---------------------------------------------------------------------------
def plot_panel(
    figs: List[Figure],
    shape: Tuple[int, int],
    title: str = "",
    save_path: Optional[str] = None,
) -> Figure:
    """Combine multiple matplotlib figures into a single panel figure.

    Each sub-figure is rasterised and embedded as an image.

    Args:
        figs (list[Figure]): Existing matplotlib Figure objects.
        shape (tuple): (rows, cols) grid layout.
        title (str): Overall panel title.
        save_path (str): Optional save path.

    Returns:
        matplotlib.Figure

    Example:
        fig = plot_panel(individual_figs, shape=(2, 2), title="Panel")
    """
    rows, cols = shape
    panel_fig, axes = plt.subplots(rows, cols,
                                   figsize=(cols * 5, rows * 5))
    axes = np.array(axes).reshape(-1)

    for i, (ax, sub_fig) in enumerate(zip(axes, figs)):
        buf = io.BytesIO()
        sub_fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        ax.imshow(np.array(img))
        ax.axis("off")

    for ax in axes[len(figs):]:
        ax.axis("off")

    if title:
        panel_fig.suptitle(title, fontsize=14, y=1.01)
    panel_fig.tight_layout()
    return _save_and_show(panel_fig, save_path)


# ---------------------------------------------------------------------------
# 13. GFLOPs bar chart
# ---------------------------------------------------------------------------
def plot_gflops_bar(
    gflops_dict: Dict[str, float],
    title: str = "",
    highlight_reduction: bool = False,
    save_path: Optional[str] = None,
) -> Figure:
    """Horizontal bar chart of GFLOPs per model variant.

    Args:
        gflops_dict (dict): {"model_name": gflops_value, ...}.
        title (str): Figure title.
        highlight_reduction (bool): If True, annotates % reduction vs first bar.
        save_path (str): Optional save path.

    Returns:
        matplotlib.Figure

    Example:
        fig = plot_gflops_bar({"DiT-XL/2 (full)": 118.64, "DiT + TDW": 82.0},
                              highlight_reduction=True)
    """
    names  = list(gflops_dict.keys())
    values = list(gflops_dict.values())
    base   = values[0]

    colors = ["#546E7A"] + ["#42A5F5"] * (len(names) - 1)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.5)

    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=11)

    if highlight_reduction and len(values) > 1:
        for i, (bar, val) in enumerate(zip(bars[1:], values[1:]), start=1):
            reduction = (1 - val / base) * 100
            ax.text(val - 3, bar.get_y() + bar.get_height() / 2,
                    f"−{reduction:.1f}%", va="center", ha="right",
                    fontsize=10, color="white", fontweight="bold")

    ax.set_xlabel("GFLOPs", fontsize=12)
    ax.set_title(title or "GFLOPs comparison", fontsize=13)
    ax.invert_yaxis()
    fig.tight_layout()
    return _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 14. Latency bar chart with error bars
# ---------------------------------------------------------------------------
def plot_latency_bar(
    means: Dict[str, float],
    stds: Dict[str, float],
    title: str = "",
    save_path: Optional[str] = None,
) -> Figure:
    """Bar chart of mean latency with ±std error bars.

    Args:
        means (dict): {"model_name": mean_ms, ...}.
        stds  (dict): {"model_name": std_ms, ...}.
        title (str): Figure title.
        save_path (str): Optional save path.

    Returns:
        matplotlib.Figure

    Example:
        fig = plot_latency_bar(
            means={"DDPM-250": 1840, "DDIM-20": 148},
            stds ={"DDPM-250": 12,   "DDIM-20": 3},
        )
    """
    names    = list(means.keys())
    mean_vals= [means[n] for n in names]
    std_vals = [stds[n]  for n in names]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    x = np.arange(len(names))
    bars = ax.bar(x, mean_vals, yerr=std_vals, capsize=6,
                  color=["#546E7A", "#42A5F5", "#66BB6A", "#FFA726"][:len(names)],
                  edgecolor="white", error_kw={"elinewidth": 2, "ecolor": "#333"})

    for bar, m, s in zip(bars, mean_vals, std_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 20,
                f"{m:.0f}ms", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(title or "Latency comparison", fontsize=13)
    fig.tight_layout()
    return _save_and_show(fig, save_path)
