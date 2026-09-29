#!/usr/bin/env python
"""Multi-GPU shard runner for the E01 probe (driver + worker in one file).

Splits the full sample list (probe A then B, interleaved for load balance)
across N GPUs. Each shard reproduces the exact single-GPU flow from
e01_probe/run.py on its slice; the driver then merges raw/controls CSVs,
restores the single-process row order, and calls report.summarize so
summary.json / interactions.csv match a single-process run on the same
hardware and library versions.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent / 'Draft-and-Refine-with-Visual-Experts'
sys.path.insert(0, str(REPO))

FIELDS = ['sample_id', 'probe', 'target_field', 'subset', 'ground_truth', 'logp_0', 'logp_1',
          'p_0', 'p_1', 'candidate_loglik_0', 'candidate_loglik_1', 'p_correct',
          'prediction', 'correct', 'cue_styles', 'model_name', 'seed', 'empty_mode', 'scoring_mode']


def build_full_samples(limit, seed):
    """Identical to the sample construction in e01_probe.run.main (probe A then B)."""
    from e01_probe.build_dataset_v2 import build_samples
    samples = []
    for probe in ('A', 'B'):
        full = build_samples(probe, max(128, limit), seed)
        build_samples(probe, limit, seed)  # validates smoke sample size
        samples.extend(full[:limit])
    return samples


def worker(args):
    from e01_probe.build_dataset_v2 import LEGENDS, write_dataset
    from e01_probe.run import question
    from process.interaction_credit import LABELS, subsets

    samples = build_full_samples(args.limit, args.seed)
    shard = samples[args.shard_index::args.num_shards]
    shard_dir = Path(args.shard_dir)
    write_dataset(shard, shard_dir / 'dataset')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from models.vlm.loader import load_qwen25_vl
    from models.vlm.qwen_vl_predict import score_binary_candidates_qwen

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, processor = load_qwen25_vl(model_name=args.model_name, device='cuda:0',
                                      cache_dir=None, attn_implementation='eager')
    model.eval()
    import importlib.metadata
    metadata = {**vars(args), 'shard_dir': str(shard_dir),
                'git_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=str(REPO), text=True).strip(),
        'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy', 'Pillow')},
        'status': 'RUNNING', 'candidate_normalization': 'log-softmax over scores for 0 and 1',
        'attention_implementation': 'eager', 'dtype': str(model.dtype)}
    (shard_dir / 'run.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    raw, controls = [], []
    with (shard_dir / 'raw_scores.csv').open('w', newline='') as raw_file, \
            (shard_dir / 'controls.csv').open('w', newline='') as control_file:
        raw_writer = csv.DictWriter(raw_file, fieldnames=FIELDS)
        control_writer = csv.DictWriter(control_file, fieldnames=FIELDS + ['control'])
        raw_writer.writeheader()
        control_writer.writeheader()

        def score(sample, target, truth, subset, prompt, images, control=None):
            result = score_binary_candidates_qwen(model, processor, prompt, images, subset)
            prediction = int(result['logp_1'] > result['logp_0'])
            row = {**result, 'sample_id': sample['sample_id'], 'probe': sample['probe'],
                   'target_field': target, 'subset': subset, 'ground_truth': truth,
                   'p_correct': result[f'p_{truth}'], 'prediction': prediction,
                   'correct': int(prediction == truth), 'cue_styles': json.dumps(sample['cue_styles'], sort_keys=True),
                   'model_name': args.model_name, 'seed': args.seed}
            if control:
                row['control'] = control
                controls.append(row)
                control_writer.writerow(row)
                control_file.flush()
            else:
                raw.append(row)
                raw_writer.writerow(row)
                raw_file.flush()

        for index, sample in enumerate(shard):
            images = {i: shard_dir / 'dataset' / f"{sample['sample_id']}_{i}.png" for i in LABELS}
            for target, truth in sample['targets'].items():
                score(sample, target, truth, 'EMPTY', question(sample, target, True), {}, 'text_only_logic')
                for subset in subsets():
                    score(sample, target, truth, subset, question(sample, target), images)
            for label in LABELS:
                prompt = (f'Image {label}: {LEGENDS[sample["cue_styles"][label]]}. '
                          f'What binary value does Image {label} encode?')
                score(sample, label, sample['bits'][label], label, prompt, images, 'single_cue')
            print(f"[{index + 1}/{len(shard)}] {sample['sample_id']}", flush=True)
    metadata['status'] = 'COMPLETE'
    metadata['raw_rows'] = len(raw)
    metadata['control_rows'] = len(controls)
    (shard_dir / 'run.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')


def merge(args, samples, output_dir):
    import shutil
    from e01_probe.report import summarize
    from process.interaction_credit import subsets

    raw, controls, manifest_rows = [], [], []
    for i in range(args.num_shards):
        shard_dir = output_dir / f'shard_{i}'
        with (shard_dir / 'raw_scores.csv').open(newline='') as stream:
            raw.extend(csv.DictReader(stream))
        with (shard_dir / 'controls.csv').open(newline='') as stream:
            controls.extend(csv.DictReader(stream))
        manifest_rows.extend(json.loads((shard_dir / 'dataset' / 'manifest.json').read_text()))

    # Restore the exact single-process row order so bootstrap draws match.
    sample_order = {s['sample_id']: i for i, s in enumerate(samples)}
    target_order = {'STEP1': 0, 'STEP2': 1, 'FINAL': 2, 'STEP': 0}
    subset_order = {s: i for i, s in enumerate(subsets())}
    label_order = {label: i for i, label in enumerate('ABCD')}

    def raw_key(row):
        return (sample_order[row['sample_id']], target_order[row['target_field']],
                subset_order[row['subset']])

    def control_key(row):
        if row['control'] == 'text_only_logic':
            return (sample_order[row['sample_id']], target_order[row['target_field']], -1)
        return (sample_order[row['sample_id']], 99, label_order[row['target_field']])

    raw.sort(key=raw_key)
    controls.sort(key=control_key)

    # Completeness: every sample/target lattice must be full and unduplicated.
    grouped = {}
    for row in raw:
        key = (row['probe'], row['sample_id'], row['target_field'])
        grouped.setdefault(key, []).append(row['subset'])
    expected = {(s['probe'], s['sample_id'], t) for s in samples for t in s['targets']}
    bad = {k: v for k, v in grouped.items() if sorted(v) != sorted(subsets())}
    if set(grouped) != expected or bad:
        print(f'Merge check failed: expected {len(expected)} lattices, '
              f'got {len(grouped)}; broken: {list(bad)[:5]}', file=sys.stderr)
        sys.exit(1)

    (output_dir / 'dataset').mkdir()
    for i in range(args.num_shards):
        for png in (output_dir / f'shard_{i}' / 'dataset').glob('*.png'):
            shutil.copy2(png, output_dir / 'dataset' / png.name)
    manifest_rows.sort(key=lambda r: r['sample_id'])
    (output_dir / 'dataset' / 'manifest.json').write_text(json.dumps(manifest_rows, indent=2) + '\n')

    for name, rows in (('raw_scores.csv', raw), ('controls.csv', controls)):
        with (output_dir / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS + ['control'])
            writer.writeheader()
            writer.writerows(rows)

    summary, credits = summarize(raw, controls, args.seed, args.bootstrap_draws,
                                 args.ability_threshold, args.loo_margin)
    with (output_dir / 'interactions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(credits[0]))
        writer.writeheader()
        writer.writerows(credits)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')

    import importlib.metadata
    metadata = {**vars(args), 'output_dir': str(output_dir),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                      cwd=str(REPO), text=True).strip(),
                'dataset_sha256': hashlib.sha256(
                    (output_dir / 'dataset' / 'manifest.json').read_bytes()).hexdigest(),
                'n_samples': len(samples), 'raw_rows': len(raw), 'control_rows': len(controls),
                'versions': {p: importlib.metadata.version(p)
                             for p in ('torch', 'transformers', 'numpy', 'Pillow')},
                'shards': [json.loads((output_dir / f'shard_{i}' / 'run.json').read_text())
                           for i in range(args.num_shards)],
                'status': 'COMPLETE', 'merged_by': 'e01_shard_runner.py',
                'candidate_normalization': 'log-softmax over scores for 0 and 1',
                'attention_implementation': 'eager'}
    (output_dir / 'run.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    print(json.dumps({'gates': summary['gates'], 'STOP_REASON': summary['STOP_REASON'],
                      'overall_status': summary['overall_status']}, indent=2))


def driver(args):
    samples = build_full_samples(args.limit, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    gpus = [g.strip() for g in args.gpus.split(',')]
    if args.num_shards > len(gpus):
        raise SystemExit(f'--num-shards {args.num_shards} exceeds --gpus {args.gpus}')
    expected_raw = sum(len(s['targets']) * 16 for s in samples)
    print(f'Launching {args.num_shards} shards over GPUs {gpus[:args.num_shards]} '
          f'({len(samples)} samples, {expected_raw} raw rows expected)', flush=True)
    procs, logs = [], []
    for i in range(args.num_shards):
        shard_dir = args.output_dir / f'shard_{i}'
        shard_dir.mkdir()
        log = (shard_dir / 'log.txt').open('w')
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker',
               '--shard-index', str(i), '--num-shards', str(args.num_shards),
               '--limit', str(args.limit), '--seed', str(args.seed),
               '--model-name', args.model_name, '--shard-dir', str(shard_dir)]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpus[i]}
        procs.append(subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT))
        logs.append(log)
    failed = [i for i, p in enumerate(procs) if p.wait() != 0]
    for log in logs:
        log.close()
    if failed:
        print(f'Shards failed: {failed}; inspect log.txt under each shard dir', file=sys.stderr)
        sys.exit(1)
    merge(args, samples, args.output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--model-name', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--probe', choices=['A', 'B', 'both'], default='both')
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--num-shards', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, default=Path('results/e01_probe_v2/full'))
    parser.add_argument('--bootstrap-draws', type=int, default=2000)
    parser.add_argument('--ability-threshold', type=float, default=.8)
    parser.add_argument('--loo-margin', type=float, default=.05)
    parser.add_argument('--shard-index', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--shard-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.ability_threshold <= 1 or args.loo_margin <= 0 or args.bootstrap_draws < 100:
        parser.error('Require 0 < ability-threshold <= 1, loo-margin > 0, bootstrap-draws >= 100')
    if args.probe != 'both':
        raise SystemExit('Shard runner always runs both probes to keep sample ids stable')
    if args.worker:
        worker(args)
    else:
        driver(args)


if __name__ == '__main__':
    main()
