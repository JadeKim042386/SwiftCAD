"""Build per-tier comparison grids for the report qualitative section.

Layout per sample row (n_views = 4):
    [GT_v0 GT_v1 GT_v2 GT_v3 | (a)_v0 (a)_v1 (a)_v2 (a)_v3 | (e)_v0 (e)_v1 (e)_v2 (e)_v3]

Each tier (top/mid/bottom) is emitted as a single PNG.
Missing (OCC-convert-failed) tiles render as a light gray placeholder with "conversion failed" label.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


BASE = "/home/work/Drawing2CAD/docs/figures/qualitative_3d"
VA = os.path.join(BASE, "variant_a_baseline_4x")
VE = os.path.join(BASE, "variant_e_alt_cross_4x")

N_VIEWS = 4
TILE = (256, 256)


def _load(path: str, tile=TILE, missing_label: str = "OCC convert FAIL") -> Tuple[np.ndarray, bool]:
    """Return (array, exists_flag)."""
    if os.path.exists(path) and os.path.getsize(path) > 200:
        img = Image.open(path).convert("RGB")
        if img.size != tile:
            img = img.resize(tile)
        return np.array(img), True
    ph = np.full((tile[1], tile[0], 3), 245, dtype=np.uint8)
    # Hatched pattern for missing
    ph[::24, :, :] = 200
    ph[:, ::24, :] = 200
    return ph, False


def compose_tier_grid(
    sample_ids: List[str],
    out_path: str,
    tier_label: str,
    scores: List[Tuple[float, float, float, float]] = None,
):
    """scores list (optional): list of (cmd_a, args_a, cmd_e, args_e) per sample for annotation."""
    n_rows = len(sample_ids)
    # 3 column blocks of N_VIEWS + 2 vertical separators
    n_cols = 3 * N_VIEWS
    fig_w = n_cols * TILE[0] / 100 + 0.4
    fig_h = n_rows * TILE[1] / 100 + 1.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    for r, sid in enumerate(sample_ids):
        # GT from variant_a (both variants render same gt)
        for v in range(N_VIEWS):
            path = os.path.join(VA, f"{sid}_gt_v{v}.png")
            arr, exists = _load(path)
            ax = axes[r, v]
            ax.imshow(arr)
            ax.set_xticks([]); ax.set_yticks([])
            if not exists:
                ax.text(0.5, 0.5, "OCC convert FAIL", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="#444",
                        bbox=dict(facecolor="white", alpha=0.8, pad=2))
            if r == 0:
                ax.set_title(f"GT v{v}", fontsize=8)

        for v in range(N_VIEWS):
            path = os.path.join(VA, f"{sid}_pred_v{v}.png")
            arr, exists = _load(path)
            ax = axes[r, N_VIEWS + v]
            ax.imshow(arr)
            ax.set_xticks([]); ax.set_yticks([])
            if not exists:
                ax.text(0.5, 0.5, "OCC convert FAIL", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="#a00",
                        bbox=dict(facecolor="white", alpha=0.85, pad=2))
            if r == 0:
                ax.set_title(f"(a) v{v}", fontsize=8)

        for v in range(N_VIEWS):
            path = os.path.join(VE, f"{sid}_pred_v{v}.png")
            arr, exists = _load(path)
            ax = axes[r, 2 * N_VIEWS + v]
            ax.imshow(arr)
            ax.set_xticks([]); ax.set_yticks([])
            if not exists:
                ax.text(0.5, 0.5, "OCC convert FAIL", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="#a00",
                        bbox=dict(facecolor="white", alpha=0.85, pad=2))
            if r == 0:
                ax.set_title(f"(e) v{v}", fontsize=8)

        # Row label (sample id) + optional scores
        if scores and r < len(scores):
            ca, aa, ce, ae = scores[r]
            label = f"{sid}\n(a) Cmd {ca:.0%}/Args {aa:.0%}\n(e) Cmd {ce:.0%}/Args {ae:.0%}"
        else:
            label = sid
        axes[r, 0].set_ylabel(label, fontsize=7.5, rotation=0, ha="right", va="center",
                               labelpad=45)

    fig.suptitle(f"{tier_label}  |  GT (left) vs (a) Baseline (center) vs (e) Alt+Cross (right)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# Tier definitions with pre-computed scores
TOP = ["00008056", "00017379"]
TOP_SCORES = [
    (1.00, 1.00, 1.00, 1.00),  # 00008056
    (1.00, 1.00, 1.00, 1.00),  # 00017379
]

MID = ["00868771", "00319566"]
MID_SCORES = [
    (0.92, 0.65, 1.00, 0.85),  # 00868771
    (1.00, 0.84, 0.58, 0.53),  # 00319566
]

BOTTOM = ["00306982", "00582849", "00625131", "00883872"]
BOTTOM_SCORES = [
    (0.07, 0.19, 0.02, 0.16),
    (0.08, 0.12, 0.03, 0.12),
    (0.07, 0.19, 0.04, 0.15),
    (0.05, 0.16, 0.05, 0.16),
]


if __name__ == "__main__":
    compose_tier_grid(TOP, os.path.join(BASE, "grid_top_tier.png"),
                      "Top Tier (Score >= 0.99)", TOP_SCORES)
    compose_tier_grid(MID, os.path.join(BASE, "grid_mid_tier.png"),
                      "Mid Tier (0.5 <= Score < 0.95)", MID_SCORES)
    compose_tier_grid(BOTTOM, os.path.join(BASE, "grid_bottom_tier.png"),
                      "Bottom Tier (Score < 0.15)", BOTTOM_SCORES)
    print("wrote", os.path.join(BASE, "grid_top_tier.png"))
    print("wrote", os.path.join(BASE, "grid_mid_tier.png"))
    print("wrote", os.path.join(BASE, "grid_bottom_tier.png"))
