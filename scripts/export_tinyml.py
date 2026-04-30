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

from spirob.model import ModelConfig, SpiRobTinyModel
from spirob.tinyml import export_onnx, export_torchscript, quantize_onnx_dynamic


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_window_steps_from_dataset(data_dir: Path) -> int:
    windows_dir = data_dir / "windows"
    first_npz = next(windows_dir.glob("window_*.npz"), None)
    if first_npz is None:
        return 80
    x = np.load(first_npz)
    return int(x["imu"].shape[0])


def make_dummy_sample(window_steps: int, n_imus: int, act_dim: int) -> dict:
    return {
        "imu": torch.zeros((1, window_steps, n_imus, 6), dtype=torch.float32),
        "mask": torch.ones((1, window_steps, n_imus), dtype=torch.float32),
        "actuator": torch.zeros((1, window_steps, act_dim), dtype=torch.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint)
    stats_path = Path(args.stats)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Infer dataset directory from norm_stats.json
    data_dir = stats_path.parent
    input_config_path = data_dir / "input_config.json"
    if not input_config_path.exists():
        raise FileNotFoundError(f"Missing input_config.json next to stats file: {input_config_path}")

    input_config = load_json(input_config_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in ckpt:
        raise KeyError("Checkpoint does not contain 'config'.")

    config = ModelConfig(**ckpt["config"])
    model = SpiRobTinyModel(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_imus = len(input_config["imu_indices"])
    act_dim = int(input_config["act_dim"])
    window_steps = infer_window_steps_from_dataset(data_dir)

    # Safety checks so export cannot silently mismatch training setup
    if config.act_dim != act_dim:
        raise ValueError(
            f"Checkpoint act_dim={config.act_dim} does not match dataset act_dim={act_dim}."
        )
    if config.num_families != int(input_config["num_families"]):
        raise ValueError(
            f"Checkpoint num_families={config.num_families} does not match dataset num_families={input_config['num_families']}."
        )

    sample = make_dummy_sample(
        window_steps=window_steps,
        n_imus=n_imus,
        act_dim=act_dim,
    )

    ts_path = export_torchscript(model, sample, outdir / "spirob_student.ts")
    onnx_path = export_onnx(model, sample, outdir / "spirob_student.onnx")
    int8_path = quantize_onnx_dynamic(onnx_path, outdir / "spirob_student_int8.onnx")

    manifest = {
        "checkpoint": str(checkpoint_path.resolve()),
        "stats": str(stats_path.resolve()),
        "data_dir": str(data_dir.resolve()),
        "input_config": str(input_config_path.resolve()),
        "model_type": ckpt.get("model_type"),
        "checkpoint_config": ckpt["config"],
        "dataset_config": {
            "imu_indices": input_config["imu_indices"],
            "actuator_columns": input_config["actuator_columns"],
            "act_dim": input_config["act_dim"],
            "num_families": input_config["num_families"],
            "family_label_mode": input_config["family_label_mode"],
        },
        "sample_shapes": {
            "imu": list(sample["imu"].shape),
            "mask": list(sample["mask"].shape),
            "actuator": list(sample["actuator"].shape),
        },
        "torchscript": str(ts_path),
        "onnx": str(onnx_path),
        "onnx_int8": str(int8_path),
    }

    with open(outdir / "export_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Export complete.")
    print("Sample shapes:")
    print("  imu:", tuple(sample["imu"].shape))
    print("  mask:", tuple(sample["mask"].shape))
    print("  actuator:", tuple(sample["actuator"].shape))
    print("Artifacts:")
    print("  TorchScript:", ts_path)
    print("  ONNX:", onnx_path)
    print("  INT8:", int8_path)


if __name__ == "__main__":
    main()