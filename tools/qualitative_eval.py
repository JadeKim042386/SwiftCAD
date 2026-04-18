"""End-to-end qualitative evaluation pipeline for Drawing2CAD.

Given one or more proj_log/<variant>/test_results/*_vec.h5 sets, this script:
  1. Renders pred and gt 3D images for each selected sample via OCC (see
     render_cad.py for the renderer).
  2. Optionally combines the per-sample multi-view PNGs into a comparison grid
     (rows = samples, columns = gt + pred views), as a single PNG per variant.
  3. Writes a JSON summary with success/failure counts and error taxonomy.

Typical usage
-------------
Run the sample-selection + rendering step for a single variant on the three
smoke-test samples:

    xvfb-run -a python tools/qualitative_eval.py \\
        --variants-root /home/work/Drawing2CAD/proj_log \\
        --variant variant_e_alt_cross_4x \\
        --output-root /home/work/Drawing2CAD/docs/figures/qualitative_3d \\
        --sample-ids 00000134 00000392 00000559

Render everything for two variants and produce a combined grid:

    xvfb-run -a python tools/qualitative_eval.py \\
        --variants-root /home/work/Drawing2CAD/proj_log \\
        --variant variant_a_baseline_4x variant_e_alt_cross_4x \\
        --output-root /home/work/Drawing2CAD/docs/figures/qualitative_3d \\
        --sample-ids 00000134 00000392 00000559 \\
        --make-grid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tools.render_cad import OCCRenderer, run as render_run  # noqa: E402


# --------------------------------------------------------------------------- #
# Grid composition
# --------------------------------------------------------------------------- #

def _load_or_placeholder(path: str, size: Tuple[int, int]) -> np.ndarray:
    if os.path.exists(path) and os.path.getsize(path) > 200:
        img = Image.open(path).convert("RGB")
        if img.size != size:
            img = img.resize(size)
        return np.array(img)
    ph = np.full((size[1], size[0], 3), 230, dtype=np.uint8)
    # write an X for missing
    ph[::32, :, :] = 180
    ph[:, ::32, :] = 180
    return ph


def compose_grid(
    output_dir: str,
    variant_label: str,
    sample_ids: List[str],
    out_path: str,
    n_views: int = 4,
    tile_size: Tuple[int, int] = (256, 256),
):
    """Compose a grid image. Row per sample. Columns = gt_v0..gt_v{n-1}, pred_v0..pred_v{n-1}."""
    n_rows = len(sample_ids)
    n_cols = 2 * n_views
    fig_w = n_cols * tile_size[0] / 100
    fig_h = n_rows * tile_size[1] / 100 + 0.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    for r, sid in enumerate(sample_ids):
        for v in range(n_views):
            for c_off, tag in [(0, "gt"), (n_views, "pred")]:
                path = os.path.join(output_dir, f"{sid}_{tag}_v{v}.png")
                ax = axes[r, c_off + v]
                ax.imshow(_load_or_placeholder(path, tile_size))
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    title = f"{tag}_v{v}"
                    ax.set_title(title, fontsize=8)
                if v == 0 and c_off == 0:
                    ax.set_ylabel(sid, fontsize=8)
    fig.suptitle(variant_label, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Success-rate aggregator (no rendering, metadata-only)
# --------------------------------------------------------------------------- #

def count_success(
    variant_dir: str,
    sample_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Walk test_results and try vec->shape conversion only. No rendering.

    Returns summary dict with OK/fail counts and error taxonomy (pred and gt).
    Much faster than full rendering — suitable for dataset-wide success rates.
    """
    import h5py
    from tools.render_cad import vec_to_shape_safe, shape_is_valid

    test_dir = os.path.join(variant_dir, "test_results")
    stats = {
        "pred": {"ok": 0, "ok_valid": 0, "fail": 0, "errors": {}},
        "gt":   {"ok": 0, "ok_valid": 0, "fail": 0, "errors": {}},
    }
    names = sorted(f for f in os.listdir(test_dir) if f.endswith("_vec.h5"))
    if sample_ids:
        names = [f"{sid}_vec.h5" for sid in sample_ids]
    if limit is not None:
        names = names[:limit]
    t0 = time.time()
    for i, name in enumerate(names):
        p = os.path.join(test_dir, name)
        if not os.path.exists(p):
            continue
        try:
            with h5py.File(p, "r") as f:
                out_vec = f["out_vec"][:]
                gt_vec = f["gt_vec"][:]
        except Exception as e:  # noqa: BLE001
            stats["pred"]["fail"] += 1
            stats["gt"]["fail"] += 1
            for tag in ("pred", "gt"):
                stats[tag]["errors"]["h5_read"] = stats[tag]["errors"].get("h5_read", 0) + 1
            continue
        for tag, vec in [("pred", out_vec), ("gt", gt_vec)]:
            shape, err = vec_to_shape_safe(vec)
            if shape is None:
                stats[tag]["fail"] += 1
                head = (err or "unknown").split(":", 1)[0]
                stats[tag]["errors"][head] = stats[tag]["errors"].get(head, 0) + 1
            else:
                stats[tag]["ok"] += 1
                if shape_is_valid(shape):
                    stats[tag]["ok_valid"] += 1
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(names)}] pred_ok={stats['pred']['ok']} "
                  f"gt_ok={stats['gt']['ok']} ({elapsed:.0f}s)", flush=True)
    stats["n_samples"] = len(names)
    stats["elapsed_sec"] = time.time() - t0
    return stats


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Qualitative eval pipeline: render + grid.")
    p.add_argument("--variants-root", default="/home/work/Drawing2CAD/proj_log")
    p.add_argument("--variant", nargs="+", required=True,
                   help="One or more variant directory names (relative to --variants-root).")
    p.add_argument("--output-root", default="/home/work/Drawing2CAD/docs/figures/qualitative_3d")
    p.add_argument("--sample-ids", nargs="*", default=None)
    p.add_argument("--n-views", type=int, default=4)
    p.add_argument("--resolution", type=int, nargs=2, default=(512, 512))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--make-grid", action="store_true",
                   help="After rendering, compose grid PNGs per variant (needs --sample-ids).")
    p.add_argument("--success-only", action="store_true",
                   help="Skip rendering; just count OCC conversion success/failure per variant.")
    p.add_argument("--log-root", default=None,
                   help="Directory for per-variant JSON logs (default: --output-root).")
    args = p.parse_args()

    log_root = args.log_root or args.output_root
    os.makedirs(log_root, exist_ok=True)

    all_summaries: Dict[str, Any] = {}
    for variant in args.variant:
        variant_dir = os.path.join(args.variants_root, variant)
        if not os.path.isdir(variant_dir):
            print(f"SKIP (not a directory): {variant_dir}")
            continue

        if args.success_only:
            print(f"\n=== success-count: {variant} ===", flush=True)
            stats = count_success(
                variant_dir=variant_dir,
                sample_ids=args.sample_ids,
                limit=args.limit,
            )
            all_summaries[variant] = stats
            log_path = os.path.join(log_root, f"success_{variant}.json")
            with open(log_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(json.dumps(stats, indent=2))
            continue

        # Full rendering
        out_dir = os.path.join(args.output_root, variant)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(log_root, f"render_{variant}.json")
        print(f"\n=== render: {variant} -> {out_dir} ===", flush=True)
        summary = render_run(
            variant_dir=variant_dir,
            output_dir=out_dir,
            sample_ids=args.sample_ids,
            n_views=args.n_views,
            resolution=tuple(args.resolution),
            log_path=log_path,
            limit=args.limit,
        )
        all_summaries[variant] = summary
        print(json.dumps(summary, indent=2))

        if args.make_grid and args.sample_ids:
            grid_path = os.path.join(args.output_root, f"grid_{variant}.png")
            compose_grid(
                output_dir=out_dir,
                variant_label=variant,
                sample_ids=args.sample_ids,
                out_path=grid_path,
                n_views=args.n_views,
            )
            print(f"grid -> {grid_path}")

    # Top-level summary
    top_log = os.path.join(log_root, "qualitative_eval_summary.json")
    with open(top_log, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nTop-level summary -> {top_log}")


if __name__ == "__main__":
    main()
