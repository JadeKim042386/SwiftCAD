"""Compute Chamfer Distance (CD) and Invalidity Ratio (IR) for SwiftCAD test outputs.

Mirrors Drawing2CAD's evaluation protocol. Consumes test.py outputs at
proj_log/<exp_name>/test_results/<data_id>_vec.h5 (each containing both
out_vec and gt_vec) and writes:
  - <output>             human-readable per-sample TSV + summary footer
  - <output>.json        machine-readable summary (consumed by eval_paper_models.py)

Two GT sources (mutually exclusive):
  --gt_pc_root <dir>   load precomputed ply (Drawing2CAD parity, recommended)
  --gt_from_h5         regenerate from gt_vec on the fly (self-contained)
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import warnings

import h5py
import numpy as np
from joblib import Parallel, delayed
from plyfile import PlyData
from scipy.spatial import cKDTree as KDTree

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cadlib.visualize import CADsolid2pc, vec2CADsolid

# trimesh.sample.sample_surface uses numpy.random; we seed np.random per-worker
# inside process_one so no explicit seed kwarg is needed.


def read_ply(path):
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
        x = np.array(plydata["vertex"]["x"])
        y = np.array(plydata["vertex"]["y"])
        z = np.array(plydata["vertex"]["z"])
    return np.stack([x, y, z], axis=1)


def normalize_pc(points):
    return points / np.max(np.abs(points))


def chamfer_dist(gt_points, gen_points):
    gen_tree = KDTree(gen_points)
    one_distances, _ = gen_tree.query(gt_points)
    gt_to_gen = float(np.mean(np.square(one_distances)))

    gt_tree = KDTree(gt_points)
    two_distances, _ = gt_tree.query(gen_points)
    gen_to_gt = float(np.mean(np.square(two_distances)))

    return gt_to_gen + gen_to_gt


def derive_seed(data_id, base_seed):
    h = int(hashlib.md5(data_id.encode("utf-8")).hexdigest()[:8], 16)
    return (h ^ int(base_seed)) & 0xFFFFFFFF


def make_gt_loader(args):
    """Return a callable: data_id, n_points -> (pc | None) using the chosen mode."""
    if args.gt_pc_root is not None:
        gt_pc_root = args.gt_pc_root

        def loader(data_id, n_points):
            ply_path = os.path.join(gt_pc_root, data_id + ".h5.ply")
            if not os.path.exists(ply_path):
                return None
            try:
                pc = read_ply(ply_path)
            except Exception:
                return None
            if pc.shape[0] >= n_points:
                idx = random.sample(range(pc.shape[0]), n_points)
                pc = pc[idx]
            return pc

        return loader

    # gt_from_h5: regenerate from gt_vec stored in test_results h5
    def loader(data_id, n_points, _src=args.src):
        h5_path = os.path.join(_src, data_id + "_vec.h5")
        try:
            with h5py.File(h5_path, "r") as fp:
                gt_vec = fp["gt_vec"][:].astype(np.float64)
            shape = vec2CADsolid(gt_vec)
            return CADsolid2pc(shape, n_points)
        except Exception:
            return None

    return loader


def process_one(path, gt_loader, normalize_gt, n_points, base_seed):
    data_id = os.path.basename(path).split("_vec")[0]
    worker_seed = derive_seed(data_id, base_seed)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    # GT
    gt_pc = gt_loader(data_id, n_points)
    if gt_pc is None:
        return {"data_id": data_id, "status": "gt_invalid", "cd": None,
                "n_pred_pts": 0, "n_gt_pts": 0}
    if normalize_gt and np.max(np.abs(gt_pc)) > 2:
        gt_pc = normalize_pc(gt_pc)

    # Pred
    try:
        with h5py.File(path, "r") as fp:
            out_vec = fp["out_vec"][:].astype(np.float64)
        out_shape = vec2CADsolid(out_vec)
        out_pc = CADsolid2pc(out_shape, n_points)
    except Exception:
        return {"data_id": data_id, "status": "pred_invalid", "cd": None,
                "n_pred_pts": 0, "n_gt_pts": int(gt_pc.shape[0])}

    if np.max(np.abs(out_pc)) > 2:
        out_pc = normalize_pc(out_pc)

    cd = chamfer_dist(gt_pc, out_pc)
    return {"data_id": data_id, "status": "ok", "cd": float(cd),
            "n_pred_pts": int(out_pc.shape[0]), "n_gt_pts": int(gt_pc.shape[0])}


TSV_COLS = ["idx", "data_id", "status", "cd", "n_pred_pts", "n_gt_pts"]


def load_resume(output_path):
    """Read processed data_ids from an existing TSV, dropping any partial last row."""
    if not os.path.exists(output_path):
        return {}, set()
    processed = {}
    with open(output_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = []
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) != len(TSV_COLS):
                continue
            rows.append(row)
    dropped_partial = 0
    # If last row was mid-write, len mismatch already filtered it. Also catch unparseable cd.
    for row in rows:
        try:
            idx = int(row[0])
            data_id = row[1]
            processed[data_id] = idx
        except ValueError:
            dropped_partial += 1
    if dropped_partial:
        print(f"[eval_cd_ir] resume: dropped {dropped_partial} partial/corrupt row(s)")
    return processed, set(processed.keys())


def run(args):
    # Resolve mode + collect paths
    paths = sorted([os.path.join(args.src, p) for p in os.listdir(args.src) if p.endswith("_vec.h5")])
    if args.num != -1:
        paths = paths[: args.num]
    if not paths:
        print(f"[eval_cd_ir] ERROR: no *_vec.h5 in {args.src}")
        return 1

    skip_ids = set()
    if args.skip_list and os.path.exists(args.skip_list):
        with open(args.skip_list) as f:
            skip_ids = {ln.strip() for ln in f if ln.strip()}
    if skip_ids:
        before = len(paths)
        paths = [p for p in paths if os.path.basename(p).split("_vec")[0] not in skip_ids]
        print(f"[eval_cd_ir] skip_list: dropped {before - len(paths)} sample(s)")

    output_path = args.output or os.path.join(os.path.dirname(args.src.rstrip("/")), "cd_ir.txt")
    json_path = output_path + ".json"

    # Resume
    processed_map, processed_ids = ({}, set())
    if args.resume:
        processed_map, processed_ids = load_resume(output_path)
        if processed_ids:
            print(f"[eval_cd_ir] resume: {len(processed_ids)} already processed; will skip")

    todo_paths = [p for p in paths if os.path.basename(p).split("_vec")[0] not in processed_ids]
    print(f"[eval_cd_ir] {len(paths)} total, {len(todo_paths)} to process, {len(processed_ids)} resumed")

    mode = "precomputed" if args.gt_pc_root is not None else "from_h5"
    settings = {
        "mode": mode,
        "gt_pc_root": args.gt_pc_root,
        "normalize_gt": bool(args.normalize_gt),
        "n_points": args.n_points,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "skip_list": args.skip_list,
    }
    banner = (
        "# eval_cd_ir.py settings: "
        f"mode={mode}, gt_pc_root={args.gt_pc_root}, normalize_gt={args.normalize_gt}, "
        f"n_points={args.n_points}, seed={args.seed}, n_jobs={args.n_jobs}, "
        f"skip_list={args.skip_list}"
    )
    ir_formula = ("(n_pred_invalid + n_gt_invalid) / N" if mode == "from_h5"
                  else "n_pred_invalid / N")
    print(banner)
    print(f"# IR formula: {ir_formula}")

    gt_loader = make_gt_loader(args)

    # Build worker that captures simple, picklable args
    def _do(p):
        return process_one(p, gt_loader, args.normalize_gt, args.n_points, args.seed)

    # Open output (line-buffered append)
    new_file = not os.path.exists(output_path) or not args.resume
    if new_file and os.path.exists(output_path):
        os.remove(output_path)
    fp = open(output_path, "a", buffering=1)
    if new_file:
        fp.write(banner + "\n")
        fp.write(f"# IR formula: {ir_formula}\n")
        fp.write("\t".join(TSV_COLS) + "\n")

    # Run
    processed_results = []  # newly computed (dicts)
    if args.n_jobs == 1:
        for i, p in enumerate(todo_paths):
            r = _do(p)
            processed_results.append(r)
            row = [str(len(processed_ids) + i), r["data_id"], r["status"],
                   f"{r['cd']:.6f}" if r["cd"] is not None else "None",
                   str(r["n_pred_pts"]), str(r["n_gt_pts"])]
            fp.write("\t".join(row) + "\n")
            if (i + 1) % 50 == 0:
                print(f"[eval_cd_ir] {i+1}/{len(todo_paths)} done")
    else:
        # joblib loky backend (default); per-task safe
        out = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=2,
                       timeout=args.timeout)(delayed(_do)(p) for p in todo_paths)
        for i, r in enumerate(out):
            processed_results.append(r)
            row = [str(len(processed_ids) + i), r["data_id"], r["status"],
                   f"{r['cd']:.6f}" if r["cd"] is not None else "None",
                   str(r["n_pred_pts"]), str(r["n_gt_pts"])]
            fp.write("\t".join(row) + "\n")

    # If resuming, also need to load existing CDs for aggregation
    all_results = processed_results
    if args.resume and processed_ids:
        with open(output_path, "r") as rf:
            reader = csv.reader(rf, delimiter="\t")
            for row in reader:
                if not row or row[0].startswith("#") or row[0] == "idx":
                    continue
                if len(row) != len(TSV_COLS):
                    continue
                data_id = row[1]
                if any(r["data_id"] == data_id for r in processed_results):
                    continue  # already in fresh results
                cd = None if row[3] == "None" else float(row[3])
                all_results.append({"data_id": data_id, "status": row[2], "cd": cd,
                                    "n_pred_pts": int(row[4]), "n_gt_pts": int(row[5])})

    # Aggregate
    n_total = len(all_results)
    valid_cds = sorted([r["cd"] for r in all_results if r["status"] == "ok" and r["cd"] is not None])
    n_ok = len(valid_cds)
    n_pred_invalid = sum(1 for r in all_results if r["status"] == "pred_invalid")
    n_gt_invalid = sum(1 for r in all_results if r["status"] == "gt_invalid")

    if mode == "from_h5":
        ir = (n_pred_invalid + n_gt_invalid) / n_total if n_total else 0.0
    else:
        ir = n_pred_invalid / n_total if n_total else 0.0

    if n_ok == 0:
        cd_mean = cd_median = cd_trim = None
        top20 = []
        warnings.warn("eval_cd_ir: n_ok == 0, no CD statistics available")
    else:
        cd_mean = float(np.mean(valid_cds))
        cd_median = float(np.median(valid_cds))
        if n_ok >= 20:
            k = max(1, int(n_ok * 0.1))
            cd_trim = float(np.mean(valid_cds[k:-k]))
        else:
            cd_trim = None
            warnings.warn(f"eval_cd_ir: n_ok={n_ok} < 20, skipping trimmed mean")
        top20 = list(reversed(valid_cds[-20:]))

    summary_lines = [
        "#" * 50,
        f"total: {n_total}",
        f"n_ok: {n_ok}",
        f"n_pred_invalid: {n_pred_invalid}",
        f"n_gt_invalid: {n_gt_invalid} (diagnostic)",
        f"IR ({ir_formula}): {ir:.4f}",
        f"cd_mean: {cd_mean if cd_mean is None else f'{cd_mean:.6f}'}",
        f"cd_median: {cd_median if cd_median is None else f'{cd_median:.6f}'}",
        f"cd_trimmed_mean: {'N/A' if cd_trim is None else f'{cd_trim:.6f}'}",
        f"top-20 worst CDs: {[round(x, 6) for x in top20]}",
    ]
    for line in summary_lines:
        fp.write(line + "\n")
        print(line)
    fp.close()

    # JSON sidecar
    summary = {
        "src": args.src,
        "settings": settings,
        "stats": {
            "total": n_total,
            "n_ok": n_ok,
            "n_pred_invalid": n_pred_invalid,
            "n_gt_invalid": n_gt_invalid,
            "ir": ir,
            "ir_formula": ir_formula,
            "cd_mean": cd_mean,
            "cd_median": cd_median,
            "cd_trimmed_mean": cd_trim,
            "top20_worst": top20,
        },
    }
    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=2)
    print(f"[eval_cd_ir] wrote {output_path} and {json_path}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="dir of *_vec.h5 from test.py (proj_log/<exp>/test_results)")

    gt_group = parser.add_mutually_exclusive_group(required=True)
    gt_group.add_argument("--gt_pc_root", type=str, default=None,
                          help="dir of precomputed <data_id>.h5.ply (Drawing2CAD parity)")
    gt_group.add_argument("--gt_from_h5", action="store_true",
                          help="regenerate GT pc from gt_vec in *_vec.h5 (self-contained)")

    parser.add_argument("--normalize_gt", action="store_true", default=False,
                        help="also apply max-abs normalization to gt_pc (default: asymmetric, only out_pc)")
    parser.add_argument("--n_points", type=int, default=2000)
    parser.add_argument("--num", type=int, default=-1, help="limit samples; -1 = all (default)")
    parser.add_argument("--n_jobs", type=int, default=8, help="joblib workers; 1 = sequential")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=120, help="per-task timeout seconds")
    parser.add_argument("--resume", action="store_true",
                        help="resume from existing output TSV, skip processed data_ids")
    parser.add_argument("--skip_list", type=str, default=None,
                        help="path to file with one data_id per line to skip")
    parser.add_argument("--output", type=str, default=None,
                        help="output TSV path (default: <src>/../cd_ir.txt). JSON sidecar at <output>.json")
    args = parser.parse_args()

    # Argparse mutex group makes one of these required, but be explicit
    if args.gt_pc_root is None and not args.gt_from_h5:
        parser.error("specify --gt_pc_root <dir> or --gt_from_h5")

    since = time.time()
    rc = run(args)
    print(f"[eval_cd_ir] elapsed: {time.time() - since:.1f}s")
    sys.exit(rc)


if __name__ == "__main__":
    main()
