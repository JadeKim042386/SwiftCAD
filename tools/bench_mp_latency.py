"""Measure inference latency of mask-predict model for various N refinement steps."""
import os
import sys
import time
import json
import numpy as np
import torch

sys.path.insert(0, '/home/work/Drawing2CAD')


def main():
    from config.config import Config
    from trainer.trainer import TrainerED
    from dataset.bi_sequence_dataset import get_dataloader

    # Build cfg by patching argv (Config uses argparse)
    argv_bak = sys.argv[:]
    sys.argv = [
        'bench', '--exp_name', 'variant_e_mask_predict',
        '--encoder_type', 'alternating', '--decoder_type', 'cross_attention',
        '--input_option', '4x', '--use_mask_predict',
        '--batch_size', '1', '--num_workers', '0',
    ]
    cfg = Config('test')
    sys.argv = argv_bak

    tr_agent = TrainerED(cfg)
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    test_loader = get_dataloader('test', cfg)

    # Collect one full batch (size=1) of inputs
    it = iter(test_loader)
    sample_batches = []
    for _ in range(50):  # 50 samples for latency measurement
        try:
            sample_batches.append(next(it))
        except StopIteration:
            break

    results = {}
    configs = [
        ('n0', 0, None),
        ('n1', 1, [0.5]),
        ('n2', 2, [0.5, 0.3]),
        ('n3', 3, [0.6, 0.4, 0.2]),
    ]

    for tag, n_steps, schedule in configs:
        # Warm-up
        with torch.no_grad():
            for data in sample_batches[:5]:
                sv = data['svg']['view'].cuda()
                sc = data['svg']['command'].cuda()
                sa = data['svg']['args'].cuda()
                _ = tr_agent.net(sv, sc, sa,
                                 n_refinement_steps=n_steps,
                                 mask_ratio_schedule=schedule)

        torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for data in sample_batches:
                sv = data['svg']['view'].cuda()
                sc = data['svg']['command'].cuda()
                sa = data['svg']['args'].cuda()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = tr_agent.net(sv, sc, sa,
                                 n_refinement_steps=n_steps,
                                 mask_ratio_schedule=schedule)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # ms

        times = np.array(times)
        results[tag] = {
            'n_refinement_steps': n_steps,
            'mask_schedule': schedule,
            'latency_mean_ms': float(times.mean()),
            'latency_std_ms': float(times.std()),
            'latency_median_ms': float(np.median(times)),
            'n_samples': len(times),
        }
        print(f"{tag}: N={n_steps}  latency = {times.mean():.3f} ± {times.std():.3f} ms (median {np.median(times):.3f})")

    out_path = '/home/work/Drawing2CAD/docs/phase4_latency.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
