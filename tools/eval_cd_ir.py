"""Compute Chamfer Distance (CD) and Invalidity Ratio (IR) for a variant's test results.

Uses DeepCAD-style pipeline:
    vec → OCC TopoDS_Shape → STL → trimesh surface sample → KDTree CD

GT point clouds are regenerated per sample from gt_vec (no precomputed cache needed).
"""
import os
import sys
import json
import argparse
import random
import glob
import numpy as np
import h5py
from scipy.spatial import cKDTree as KDTree
from joblib import Parallel, delayed

sys.path.insert(0, '/home/work/Drawing2CAD')
sys.path.insert(0, '/home/work/Drawing2CAD/cadlib_deepcad')

import trimesh
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import write_stl_file
from cadlib_deepcad.visualize import vec2CADsolid


def vec_to_pc(vec, n_points, tmp_dir):
    """vec → TopoDS_Shape → STL → point cloud (N,3). Returns (pc or None, error_tag)."""
    try:
        vec = np.asarray(vec).astype(float)
        shape = vec2CADsolid(vec)
    except Exception as e:
        return None, f"convert:{type(e).__name__}"
    try:
        if not BRepCheck_Analyzer(shape).IsValid():
            return None, "invalid_brep"
    except Exception as e:
        return None, f"check:{type(e).__name__}"

    stl_path = os.path.join(tmp_dir, f"{os.getpid()}_{id(vec)}.stl")
    try:
        write_stl_file(shape, stl_path)
        mesh = trimesh.load(stl_path, force='mesh')
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None, "empty_mesh"
        pc, _ = trimesh.sample.sample_surface(mesh, n_points)
        pc = np.asarray(pc, dtype=np.float64)
    except Exception as e:
        return None, f"mesh:{type(e).__name__}"
    finally:
        if os.path.exists(stl_path):
            try:
                os.remove(stl_path)
            except Exception:
                pass

    if np.max(np.abs(pc)) > 2.0:
        scale = np.max(np.abs(pc))
        pc = pc / scale
    return pc, None


def chamfer_dist(a, b):
    t_a = KDTree(a)
    t_b = KDTree(b)
    d_ab, _ = t_b.query(a)
    d_ba, _ = t_a.query(b)
    return float(np.mean(d_ab ** 2) + np.mean(d_ba ** 2))


def process_one(h5_path, n_points, tmp_dir):
    with h5py.File(h5_path, 'r') as f:
        out_vec = f['out_vec'][:]
        gt_vec = f['gt_vec'][:]
    pred_pc, pred_err = vec_to_pc(out_vec, n_points, tmp_dir)
    gt_pc, gt_err = vec_to_pc(gt_vec, n_points, tmp_dir)
    if pred_pc is None or gt_pc is None:
        return {
            'sample': os.path.basename(h5_path),
            'cd': None,
            'pred_err': pred_err,
            'gt_err': gt_err,
        }
    cd = chamfer_dist(gt_pc, pred_pc)
    return {'sample': os.path.basename(h5_path), 'cd': cd, 'pred_err': None, 'gt_err': None}


def run(test_dir, n_points=2000, n_samples=None, n_jobs=8, seed=0):
    files = sorted(glob.glob(os.path.join(test_dir, '*.h5')))
    if n_samples is not None and n_samples < len(files):
        rng = random.Random(seed)
        files = rng.sample(files, n_samples)
        files.sort()

    tmp_dir = f"/tmp/cd_eval_{os.getpid()}"
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"[{test_dir}] {len(files)} samples, n_points={n_points}, n_jobs={n_jobs}")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_one)(f, n_points, tmp_dir) for f in files
    )

    cds = [r['cd'] for r in results if r['cd'] is not None]
    n_valid = len(cds)
    n_total = len(results)
    n_invalid = n_total - n_valid

    err_counts = {}
    for r in results:
        if r['cd'] is None:
            tag = f"pred:{r['pred_err']}" if r['pred_err'] else f"gt:{r['gt_err']}"
            err_counts[tag] = err_counts.get(tag, 0) + 1

    summary = {
        'n_total': n_total,
        'n_valid': n_valid,
        'n_invalid': n_invalid,
        'ir': n_invalid / n_total if n_total else 0.0,
        'cd_mean': float(np.mean(cds)) if cds else None,
        'cd_median': float(np.median(cds)) if cds else None,
        'cd_trimmed_mean': float(np.mean(sorted(cds)[int(n_valid*0.1):-int(n_valid*0.1)])) if n_valid > 20 else None,
        'err_breakdown': err_counts,
    }
    return summary, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-dirs', nargs='+', required=True, help='list of test_results directories')
    ap.add_argument('--labels', nargs='+', required=True, help='labels matching --test-dirs')
    ap.add_argument('--n-points', type=int, default=2000)
    ap.add_argument('--n-samples', type=int, default=2000, help='subset size (default 2000, use -1 for all)')
    ap.add_argument('--n-jobs', type=int, default=8)
    ap.add_argument('--out', type=str, default='/home/work/Drawing2CAD/docs/phase4_cd_ir.json')
    args = ap.parse_args()

    assert len(args.test_dirs) == len(args.labels)
    n_samples = None if args.n_samples < 0 else args.n_samples

    overall = {}
    for label, td in zip(args.labels, args.test_dirs):
        print(f'\n===== {label} =====')
        summary, per_sample = run(td, args.n_points, n_samples, args.n_jobs)
        overall[label] = summary
        print(json.dumps(summary, indent=2))

    with open(args.out, 'w') as f:
        json.dump(overall, f, indent=2)
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
