"""Render Drawing2CAD predictions (out_vec / gt_vec) to 3D PNG images.

Pipeline:
  1. Read proj_log/<variant>/test_results/<sample_id>_vec.h5
  2. Convert out_vec / gt_vec -> OCC TopoDS_Shape via DeepCAD's vec2CADsolid
  3. Validate with BRepCheck_Analyzer
  4. Render multi-view PNGs via OCC Viewer3d (requires xvfb)

Offscreen note: OCC's Viewer3d requires an X server. Run this script inside
`xvfb-run -a python tools/render_cad.py ...` on headless hosts.

Environment: conda env `deepcad_viz` (pythonocc-core=7.5.1, numpy<1.24).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

# Ensure DeepCAD cadlib copy is importable before any third-party import
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import h5py

# Force headless matplotlib (sketch.py also sets Agg, but be safe)
import matplotlib
matplotlib.use("Agg")

from cadlib_deepcad.visualize import vec2CADsolid  # noqa: E402
from OCC.Core.BRepCheck import BRepCheck_Analyzer  # noqa: E402
from OCC.Core.TopoDS import TopoDS_Shape  # noqa: E402
from OCC.Core.gp import gp_Pnt, gp_Dir  # noqa: E402
from OCC.Display.OCCViewer import Viewer3d  # noqa: E402


# --------------------------------------------------------------------------- #
# vec -> shape
# --------------------------------------------------------------------------- #

def vec_to_shape_safe(vec: np.ndarray) -> Tuple[Optional[TopoDS_Shape], Optional[str]]:
    """Convert a (seq_len, 17) vec to TopoDS_Shape. Returns (shape, None) or (None, err).

    The second element encodes the short error tag used for failure taxonomy.
    """
    if vec is None or vec.size == 0:
        return None, "EmptyVec"
    try:
        shape = vec2CADsolid(vec)
    except IndexError as e:
        return None, f"IndexError:{str(e)[:60]}"
    except ValueError as e:
        return None, f"ValueError:{str(e)[:60]}"
    except AssertionError as e:
        return None, f"AssertionError:{str(e)[:60]}"
    except NotImplementedError as e:
        return None, f"NotImplementedError:{str(e)[:60]}"
    except RuntimeError as e:
        return None, f"RuntimeError:{str(e)[:60]}"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}:{str(e)[:60]}"

    if shape is None:
        return None, "NullShape"
    try:
        if shape.IsNull():
            return None, "ShapeIsNull"
    except Exception:  # noqa: BLE001
        pass
    return shape, None


def shape_is_valid(shape: TopoDS_Shape) -> bool:
    try:
        return bool(BRepCheck_Analyzer(shape).IsValid())
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

# Canonical view directions (proj vector, up vector) in world coordinates.
# These approximate isometric-like shots so the user sees structure.
VIEW_DIRS: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = [
    ((1.0,  1.0,  1.0),  (0.0, 0.0, 1.0)),   # iso NE-top
    ((-1.0, 1.0,  1.0),  (0.0, 0.0, 1.0)),   # iso NW-top
    ((-1.0, -1.0, 1.0),  (0.0, 0.0, 1.0)),   # iso SW-top
    ((1.0, -1.0, -0.5),  (0.0, 0.0, 1.0)),   # lower-front
]


class OCCRenderer:
    """Reusable OCC viewer wrapper; expensive to construct, cheap to reuse."""

    def __init__(self, resolution: Tuple[int, int] = (512, 512)):
        self.resolution = resolution
        self.viewer = Viewer3d()
        self.viewer.Create(phong_shading=True, create_default_lights=True)
        self.viewer.SetSize(*resolution)
        self.viewer.SetModeShaded()
        # Pale background so white geometry is visible
        try:
            self.viewer.set_bg_gradient_color([240, 240, 245], [200, 200, 210])
        except Exception:  # noqa: BLE001
            pass

    def _set_direction(self, proj_xyz, up_xyz):
        view = self.viewer.View
        view.SetProj(float(proj_xyz[0]), float(proj_xyz[1]), float(proj_xyz[2]))
        try:
            view.SetUp(float(up_xyz[0]), float(up_xyz[1]), float(up_xyz[2]))
        except Exception:  # noqa: BLE001
            pass

    def render_views(
        self,
        shape: TopoDS_Shape,
        out_paths: List[str],
        n_views: int = 4,
    ) -> List[str]:
        """Render n_views PNG images of `shape`. Returns list of written paths.

        out_paths length must equal n_views.
        """
        assert len(out_paths) == n_views, "out_paths must have n_views entries"
        viewer = self.viewer
        viewer.EraseAll()
        viewer.DisplayShape(shape, update=False)
        written: List[str] = []
        for i in range(n_views):
            proj, up = VIEW_DIRS[i % len(VIEW_DIRS)]
            self._set_direction(proj, up)
            viewer.View.FitAll()
            viewer.View.ZFitAll()
            viewer.Repaint()
            ok = viewer.View.Dump(out_paths[i])
            # Some OCC versions return None; check file existence / size
            if not os.path.exists(out_paths[i]) or os.path.getsize(out_paths[i]) < 200:
                raise RuntimeError(f"Viewer3d.Dump produced empty file: {out_paths[i]} (ok={ok})")
            written.append(out_paths[i])
        viewer.EraseAll()
        return written


# --------------------------------------------------------------------------- #
# Sample-level driver
# --------------------------------------------------------------------------- #

def render_sample(
    h5_path: str,
    output_dir: str,
    renderer: OCCRenderer,
    n_views: int = 4,
    skip_if_exists: bool = True,
) -> Dict[str, Any]:
    """Render pred/gt for one sample. Returns a result dict with status/errors."""
    sample_id = os.path.basename(h5_path).split("_")[0]
    os.makedirs(output_dir, exist_ok=True)

    try:
        with h5py.File(h5_path, "r") as f:
            out_vec = f["out_vec"][:]
            gt_vec = f["gt_vec"][:]
    except Exception as e:  # noqa: BLE001
        return {
            "sample_id": sample_id,
            "h5_path": h5_path,
            "pred": {"status": "error", "error": f"h5_read:{e}"},
            "gt": {"status": "error", "error": f"h5_read:{e}"},
        }

    result: Dict[str, Any] = {"sample_id": sample_id, "h5_path": h5_path}
    for tag, vec in [("pred", out_vec), ("gt", gt_vec)]:
        paths = [
            os.path.join(output_dir, f"{sample_id}_{tag}_v{i}.png")
            for i in range(n_views)
        ]
        # skip if all views already exist
        if skip_if_exists and all(os.path.exists(p) and os.path.getsize(p) > 200 for p in paths):
            result[tag] = {"status": "cached", "paths": paths, "valid": None}
            continue

        shape, err = vec_to_shape_safe(vec)
        if shape is None:
            result[tag] = {"status": "convert_failed", "error": err}
            continue

        valid = shape_is_valid(shape)
        try:
            renderer.render_views(shape, paths, n_views=n_views)
            result[tag] = {"status": "ok", "paths": paths, "valid": valid}
        except Exception as e:  # noqa: BLE001
            result[tag] = {
                "status": "render_failed",
                "error": f"{type(e).__name__}:{str(e)[:100]}",
                "valid": valid,
            }
    return result


# --------------------------------------------------------------------------- #
# Dataset-level aggregate
# --------------------------------------------------------------------------- #

def iter_h5_files(test_dir: str, sample_ids: Optional[List[str]] = None):
    if sample_ids:
        for sid in sample_ids:
            p = os.path.join(test_dir, f"{sid}_vec.h5")
            if os.path.exists(p):
                yield p
            else:
                # Still yield so caller can record "missing"
                yield p
        return
    for name in sorted(os.listdir(test_dir)):
        if name.endswith("_vec.h5"):
            yield os.path.join(test_dir, name)


def run(
    variant_dir: str,
    output_dir: str,
    sample_ids: Optional[List[str]] = None,
    n_views: int = 4,
    resolution: Tuple[int, int] = (512, 512),
    log_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    test_dir = os.path.join(variant_dir, "test_results")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"test_results not found under {variant_dir}")

    os.makedirs(output_dir, exist_ok=True)
    renderer = OCCRenderer(resolution=resolution)

    results: List[Dict[str, Any]] = []
    pred_stats = {"ok": 0, "cached": 0, "convert_failed": 0, "render_failed": 0}
    gt_stats = {"ok": 0, "cached": 0, "convert_failed": 0, "render_failed": 0}
    pred_err_counter: Dict[str, int] = {}
    gt_err_counter: Dict[str, int] = {}

    t0 = time.time()
    n = 0
    for h5_path in iter_h5_files(test_dir, sample_ids):
        if limit is not None and n >= limit:
            break
        n += 1
        if not os.path.exists(h5_path):
            results.append({
                "sample_id": os.path.basename(h5_path).split("_")[0],
                "h5_path": h5_path,
                "pred": {"status": "missing"},
                "gt": {"status": "missing"},
            })
            continue
        res = render_sample(h5_path, output_dir, renderer, n_views=n_views)
        results.append(res)
        for tag, bucket, err_counter in [
            ("pred", pred_stats, pred_err_counter),
            ("gt", gt_stats, gt_err_counter),
        ]:
            st = res.get(tag, {}).get("status", "missing")
            if st in bucket:
                bucket[st] += 1
            else:
                bucket[st] = bucket.get(st, 0) + 1
            if st in ("convert_failed", "render_failed"):
                err = res[tag].get("error", "unknown")
                tag_head = err.split(":", 1)[0]
                err_counter[tag_head] = err_counter.get(tag_head, 0) + 1
        if n % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{n}] pred_ok={pred_stats['ok']} gt_ok={gt_stats['ok']} "
                  f"pred_fail={pred_stats['convert_failed']}+{pred_stats['render_failed']} "
                  f"gt_fail={gt_stats['convert_failed']}+{gt_stats['render_failed']} "
                  f"({elapsed:.0f}s)", flush=True)

    summary = {
        "variant_dir": variant_dir,
        "output_dir": output_dir,
        "n_samples": n,
        "n_views": n_views,
        "resolution": list(resolution),
        "elapsed_sec": time.time() - t0,
        "pred": {**pred_stats, "error_taxonomy": pred_err_counter},
        "gt": {**gt_stats, "error_taxonomy": gt_err_counter},
    }

    if log_path:
        with open(log_path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)

    return summary


def main():
    p = argparse.ArgumentParser(description="Render Drawing2CAD pred/gt vec as 3D PNGs via OCC.")
    p.add_argument("--variant-dir", required=True,
                   help="e.g. /home/work/Drawing2CAD/proj_log/variant_e_alt_cross_4x")
    p.add_argument("--output-dir", required=True,
                   help="Where PNGs are written.")
    p.add_argument("--log-path", default=None, help="JSON summary+per-sample log output.")
    p.add_argument("--sample-ids", nargs="*", default=None,
                   help="Optional subset of sample ids (e.g. 00000134 00000392). "
                        "If omitted, renders all samples in test_results/.")
    p.add_argument("--n-views", type=int, default=4)
    p.add_argument("--resolution", type=int, nargs=2, default=(512, 512),
                   help="width height")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N samples (debug).")
    args = p.parse_args()

    summary = run(
        variant_dir=args.variant_dir,
        output_dir=args.output_dir,
        sample_ids=args.sample_ids,
        n_views=args.n_views,
        resolution=tuple(args.resolution),
        log_path=args.log_path,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
