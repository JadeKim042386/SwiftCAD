"""Evaluate mask-predict test results (variant_e_mask_predict_n{0,1,2,3})."""
import os
import sys
import json
import numpy as np
import h5py
from tqdm import tqdm

sys.path.insert(0, '/home/work/Drawing2CAD')
from config.macro import (
    CAD_EXT_IDX, CAD_LINE_IDX, CAD_ARC_IDX, CAD_CIRCLE_IDX,
    CAD_N_ARGS_EXT, CAD_N_ARGS_PLANE, CAD_N_ARGS_TRANS, CAD_N_ARGS_EXT_PARAM,
)


def calculate_accuracy(out_vec, gt_vec, tolerance=3):
    pred_cmd = out_vec[:, 0]
    gt_cmd = gt_vec[:, 0]
    pred_args = out_vec[:, 1:]
    gt_args = gt_vec[:, 1:]

    cmd_match = (pred_cmd == gt_cmd)
    cmd_acc = float(np.mean(cmd_match))
    args_close = (np.abs(gt_args - pred_args) <= tolerance)

    def idx(cmd_type):
        return np.where((gt_cmd == cmd_type) & cmd_match)[0]

    ext_pos = idx(CAD_EXT_IDX)
    line_pos = idx(CAD_LINE_IDX)
    arc_pos = idx(CAD_ARC_IDX)
    circle_pos = idx(CAD_CIRCLE_IDX)

    r = {}
    r['line'] = args_close[line_pos][:, :2].flatten().astype(np.int32) if len(line_pos) else np.array([])
    r['arc'] = args_close[arc_pos][:, :4].flatten().astype(np.int32) if len(arc_pos) else np.array([])
    r['circle'] = args_close[circle_pos][:, [0, 1, 4]].flatten().astype(np.int32) if len(circle_pos) else np.array([])
    if len(ext_pos):
        ext_args = args_close[ext_pos][:, -CAD_N_ARGS_EXT:]
        r['plane'] = ext_args[:, :CAD_N_ARGS_PLANE].flatten().astype(np.int32)
        r['trans'] = ext_args[:, CAD_N_ARGS_PLANE:CAD_N_ARGS_PLANE + CAD_N_ARGS_TRANS].flatten().astype(np.int32)
        r['extent'] = ext_args[:, -CAD_N_ARGS_EXT_PARAM].flatten().astype(np.int32)
    else:
        r['plane'] = r['trans'] = r['extent'] = np.array([])
    return cmd_acc, r


def eval_dir(test_dir, tolerance=3):
    if not os.path.isdir(test_dir):
        return None
    files = [f for f in os.listdir(test_dir) if f.endswith('.h5')]
    if not files:
        return None

    total_cmd = 0.0
    bucket = {k: [] for k in ['line', 'arc', 'circle', 'plane', 'trans', 'extent']}
    for fn in tqdm(files, desc=os.path.basename(test_dir)):
        with h5py.File(os.path.join(test_dir, fn), 'r') as f:
            out_vec = f['out_vec'][:]
            gt_vec = f['gt_vec'][:]
        cmd_acc, r = calculate_accuracy(out_vec, gt_vec, tolerance)
        total_cmd += cmd_acc
        for k, v in r.items():
            if len(v):
                bucket[k].append(v)

    cmd_acc = total_cmd / len(files)
    args = {}
    for k, v in bucket.items():
        args[k] = float(np.mean(np.concatenate(v))) if len(v) else None
    valid = [v for v in args.values() if v is not None]
    args_avg = float(np.mean(valid)) if valid else None
    return {
        'n_files': len(files),
        'cmd_acc': cmd_acc,
        'args': args,
        'args_avg': args_avg,
    }


def main():
    base = '/home/work/Drawing2CAD/proj_log/variant_e_mask_predict'
    results = {}
    for n in [0, 1, 2, 3]:
        d = f'{base}/test_results_n{n}'
        print(f'\n=== N={n} -> {d} ===')
        res = eval_dir(d)
        if res is None:
            print('  (not found)')
            continue
        results[f'n{n}'] = res
        print(f"  Cmd Acc  : {res['cmd_acc']*100:.2f}%")
        print(f"  Args Avg : {res['args_avg']*100:.2f}%")
        for k, v in res['args'].items():
            if v is not None:
                print(f"    {k:7}: {v*100:.2f}%")

    out_path = '/home/work/Drawing2CAD/docs/phase4_accuracy.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
