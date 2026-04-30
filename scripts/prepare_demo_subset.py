#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import random
import shutil


def read_manifest(path: Path) -> list[dict]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description='Copy a tiny processed subset into data/demo for a lightweight GitHub demo.')
    ap.add_argument('--source-data-dir', required=True, help='Processed dataset directory containing manifest.jsonl and windows/')
    ap.add_argument('--outdir', required=True, help='Destination demo folder, e.g. data/demo')
    ap.add_argument('--num-samples', type=int, default=8)
    ap.add_argument('--split', default='test', help='Preferred split to sample from: test, val, train, unseen-object, unseen-geometry, or all')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    src = Path(args.source_data_dir)
    dst = Path(args.outdir)
    dst_windows = dst / 'windows'
    dst_windows.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(src / 'manifest.jsonl')
    if args.split not in {'all', '*'}:
        manifest = [m for m in manifest if m.get('split') == args.split]
    if not manifest:
        raise ValueError('No manifest entries matched the requested split.')

    rng = random.Random(args.seed)
    chosen = manifest if len(manifest) <= args.num_samples else rng.sample(manifest, args.num_samples)
    new_manifest = []
    for item in chosen:
        src_path = Path(item['path'])
        new_path = dst_windows / src_path.name
        shutil.copy2(src_path, new_path)
        item2 = dict(item)
        item2['path'] = str(new_path.resolve())
        new_manifest.append(item2)

    with open(dst / 'manifest.jsonl', 'w', encoding='utf-8') as f:
        for item in new_manifest:
            f.write(json.dumps(item) + '\n')

    for extra in ['norm_stats.json', 'input_config.json', 'label_maps.json']:
        src_extra = src / extra
        if src_extra.exists():
            shutil.copy2(src_extra, dst / extra)

    summary = {
        'source_data_dir': str(src.resolve()),
        'num_samples': len(new_manifest),
        'split': args.split,
        'windows_dir': str(dst_windows.resolve()),
    }
    with open(dst / 'demo_subset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
