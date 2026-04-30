#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

import numpy as np
import torch

from spirob.data import load_stats
from spirob.labels import ID_TO_ACTION
from spirob.model import ModelConfig, SpiRobTinyModel


def resolve_family_labels(input_config: dict, expected_num_families: int | None = None) -> list[str]:
    labels = input_config.get("family_labels") or input_config.get("coarse_shape_labels")
    if isinstance(labels, list) and labels:
        labels = [str(x) for x in labels]
        if expected_num_families is None or len(labels) == expected_num_families:
            return labels

    mode = input_config.get("family_label_mode")
    if mode == "exact7":
        labels = ["sphere", "cylinder", "capsule", "box", "flat", "branched", "irregular"]
    elif mode == "coarse5":
        labels = ["round", "elongated", "square", "flat", "irregular"]
    else:
        labels = ["round", "elongated", "square"]

    if expected_num_families is not None:
        return labels[:expected_num_families]
    return labels


def load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def maybe_load_stats(path: str | None):
    if not path:
        class Dummy:
            imu_mean = np.zeros((6,), dtype=np.float32)
            imu_std = np.ones((6,), dtype=np.float32)
            act_mean = np.zeros((2,), dtype=np.float32)
            act_std = np.ones((2,), dtype=np.float32)
        return Dummy()
    return load_stats(path)


def main() -> None:
    ap = argparse.ArgumentParser(description='Run one inference example on a single window_*.npz file.')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--window', required=True, help='Path to a single window_*.npz file')
    ap.add_argument('--stats', default='', help='Optional path to norm_stats.json')
    ap.add_argument('--input-config', default='', help='Optional path to input_config.json for family names')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = ap.parse_args()

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SpiRobTinyModel(ModelConfig(**ckpt['config'])).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    arr = np.load(args.window, allow_pickle=True)
    imu = arr['imu'].astype(np.float32)
    mask = arr['mask'].astype(np.float32)
    actuator = arr['actuator'].astype(np.float32)

    stats = maybe_load_stats(args.stats)
    imu = (imu - stats.imu_mean.reshape(1, 1, -1)) / np.maximum(stats.imu_std.reshape(1, 1, -1), 1e-6)
    actuator = (actuator - stats.act_mean.reshape(1, -1)) / np.maximum(stats.act_std.reshape(1, -1), 1e-6)

    imu_t = torch.from_numpy(imu[None, ...]).to(device)
    mask_t = torch.from_numpy(mask[None, ...]).to(device)
    act_t = torch.from_numpy(actuator[None, ...]).to(device)

    with torch.no_grad():
        out = model(imu_t, mask_t, act_t)

    input_config = {}
    if args.input_config:
        input_config = load_json(Path(args.input_config))
    elif isinstance(ckpt.get('input_config'), dict):
        input_config = ckpt['input_config']
    family_labels = resolve_family_labels(input_config, expected_num_families=int(out['family_logits'].shape[-1]))

    action_id = int(torch.argmax(out['action_logits'], dim=-1).item())
    family_id = int(torch.argmax(out['family_logits'], dim=-1).item())

    result = {
        'window_path': str(Path(args.window).resolve()),
        'success_prob': float(out['success_prob'].item()),
        'quality_pred': float(out['quality'].item()),
        'size_pred': float(out['size'].item()),
        'confidence_pred': float(out['confidence'].item()),
        'action_pred_id': action_id,
        'action_pred_name': ID_TO_ACTION.get(action_id, f'class_{action_id}'),
        'family_pred_id': family_id,
        'family_pred_name': family_labels[family_id] if 0 <= family_id < len(family_labels) else f'class_{family_id}',
        'trust_pred': [float(x) for x in out['trust'][0].detach().cpu().numpy().tolist()],
    }

    ground_truth = {}
    for key in ['success', 'quality', 'size', 'family_id', 'action_id', 'phase_id']:
        if key in arr:
            value = np.asarray(arr[key])
            ground_truth[key] = value.item() if value.ndim == 0 else value.tolist()
    if ground_truth:
        result['ground_truth'] = ground_truth

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
