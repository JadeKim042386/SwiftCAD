"""Precompute ground-truth point clouds (.ply) for CD/IR evaluation.

Two equivalent input sources (mutually exclusive):

  (A) `--src data/cad_vec`  + `--split_json` + `--phase test`
      Read CAD vectors from the dataset (`cad_vec/<id>.h5` 'vec' field).
      This is the canonical Drawing2CAD path, but requires the full
      CAD-VGDrawing release.

  (B) `--src_test_results proj_log/<exp>/test_results`
      Read GT vectors from the `gt_vec` field saved by `test.py`. The
      content is byte-equivalent to (A) after EOS truncation (verified
      empirically), and avoids requiring the full dataset.

Output: <output>/<data_id>.h5.ply (suffix kept for parity with
Drawing2CAD's cad_vec_pc layout).
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
from joblib import Parallel, delayed
from plyfile import PlyData, PlyElement

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cadlib.visualize import CADsolid2pc, vec2CADsolid


def write_ply(points, filename):
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    el = PlyElement.describe(vertex, "vertex", comments=["vertices"])
    with open(filename, mode="wb") as f:
        PlyData([el], text=False).write(f)


def process_one(h5_path, output_dir, n_points, vec_field, data_id):
    save_path = os.path.join(output_dir, data_id + ".h5.ply")
    if os.path.exists(save_path):
        return ("skip", data_id)

    try:
        with h5py.File(h5_path, "r") as fp:
            vec = fp[vec_field][:].astype(np.float64)
        shape = vec2CADsolid(vec)
        pc = CADsolid2pc(shape, n_points)
    except Exception as e:
        return ("fail", f"{data_id}: {type(e).__name__}: {str(e)[:80]}")

    write_ply(pc, save_path)
    return ("ok", data_id)


def collect_from_dataset(src, split_json, phase):
    """(path, data_id) tuples for the dataset-style cad_vec/<id>.h5 layout."""
    with open(split_json, "r") as f:
        items = json.load(f).get(phase, [])
    out = []
    for sid in items:
        p = os.path.join(src, sid + ".h5")
        # data_id = file basename (e.g., "00359481") to match test.py's save naming
        did = os.path.basename(sid).split(".")[0]
        out.append((p, did))
    return out


def collect_from_test_results(src):
    """(path, data_id) tuples for test_results/<id>_vec.h5 layout."""
    out = []
    for f in sorted(os.listdir(src)):
        if not f.endswith("_vec.h5"):
            continue
        did = f.split("_vec")[0]
        out.append((os.path.join(src, f), did))
    return out


def main():
    parser = argparse.ArgumentParser()
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--src", default=None,
                           help="cad_vec directory containing <data_id>.h5 (dataset path; reads 'vec')")
    src_group.add_argument("--src_test_results", default=None,
                           help="test_results dir containing <id>_vec.h5 (reads 'gt_vec'; equivalent to --src after EOS truncation)")

    parser.add_argument("--split_json", help="train_val_test_split.json (only required with --src)")
    parser.add_argument("--phase", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output", required=True, help="output dir for <data_id>.h5.ply")
    parser.add_argument("--n_points", type=int, default=2000,
                        help="ply size (matches Drawing2CAD's collect_gen_pc.py default)")
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()

    if args.src is not None:
        if not args.split_json:
            parser.error("--split_json is required with --src")
        paths_ids = collect_from_dataset(args.src, args.split_json, args.phase)
        vec_field = "vec"
        src_descr = f"dataset cad_vec ({args.src}, phase={args.phase})"
    else:
        paths_ids = collect_from_test_results(args.src_test_results)
        vec_field = "gt_vec"
        src_descr = f"test_results ({args.src_test_results})"

    os.makedirs(args.output, exist_ok=True)
    print(f"[precompute_pc] {len(paths_ids)} h5 files from {src_descr}")
    print(f"[precompute_pc] reading field '{vec_field}', writing to {args.output}, n_points={args.n_points}")

    results = Parallel(n_jobs=args.n_jobs, verbose=2)(
        delayed(process_one)(p, args.output, args.n_points, vec_field, did)
        for p, did in paths_ids
    )

    n_ok = sum(1 for r in results if r[0] == "ok")
    n_skip = sum(1 for r in results if r[0] == "skip")
    n_fail = sum(1 for r in results if r[0] == "fail")
    print(f"[precompute_pc] ok={n_ok}  skip={n_skip}  fail={n_fail}")
    if n_fail and n_fail < 30:
        print("[precompute_pc] failures:")
        for r in results:
            if r[0] == "fail":
                print(f"  - {r[1]}")


if __name__ == "__main__":
    main()
