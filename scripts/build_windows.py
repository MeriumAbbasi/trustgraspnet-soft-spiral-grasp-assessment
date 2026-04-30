#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import re
from typing import List
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from spirob.labels import (
    ACTION_TO_ID,
    PHASE_TO_ID,
    add_trial_features,
    oracle_intervention_label,
    window_phase_label,
    compute_trial_summary,
)
from spirob.xml_utils import get_family_label_info, parse_imu_layout
from spirob.data import compute_stats_from_manifest, save_stats


DEFAULT_8_IMUS = [0, 3, 7, 10, 12, 16, 18, 20]

ACTUATOR_PRESETS = {
    "full": [
        "ctrl_1",
        "ctrl_2",
        "actuator_force_1",
        "actuator_force_2",
        "tendon_length_1",
        "tendon_length_2",
        "tendon_velocity_1",
        "tendon_velocity_2",
    ],
    "disp_only": [
        "ctrl_1",
        "ctrl_2",
    ],
    "tendon_length_only": [
        "tendon_length_1",
        "tendon_length_2",
    ],
    "disp_and_tendon_length": [
        "ctrl_1",
        "ctrl_2",
        "tendon_length_1",
        "tendon_length_2",
    ],
    "tendon_velocity_only": [
        "tendon_velocity_1",
        "tendon_velocity_2",
    ],
}


def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def resolve_imu_indices(available: list[int], imu_preset: str, imu_indices_arg: str | None) -> list[int]:
    if imu_preset == "all":
        chosen = list(available)
    elif imu_preset == "safe8":
        chosen = [i for i in DEFAULT_8_IMUS if i in available]
    elif imu_preset == "custom":
        requested = parse_int_list(imu_indices_arg)
        if not requested:
            raise ValueError("--imu-preset custom requires --imu-indices")
        missing = [i for i in requested if i not in available]
        if missing:
            raise ValueError(f"Requested IMUs not available in XML/data: {missing}")
        chosen = requested
    else:
        raise ValueError(f"Unknown imu preset: {imu_preset}")

    if not chosen:
        raise ValueError("No IMUs selected.")
    return chosen




def infer_available_imu_indices_from_raw(raw_dir: Path) -> list[int]:
    trial_dirs = sorted((raw_dir / "trials").glob("trial_*"))
    for trial_path in trial_dirs:
        ts_path = trial_path / "timeseries.csv"
        if not ts_path.exists():
            continue
        df = pd.read_csv(ts_path, nrows=1)
        found = set()
        for col in df.columns:
            m = re.match(r"imu_(?:gyro|acc)_B(\d+)_", str(col))
            if m:
                found.add(int(m.group(1)))
        if found:
            return sorted(found)
    raise FileNotFoundError(
        f"Could not infer IMU indices from raw data under {raw_dir}. Provide --base-xml or make sure trials/trial_*/timeseries.csv exists."
    )

def resolve_actuator_columns(actuator_preset: str, actuator_columns_arg: str | None) -> list[str]:
    if actuator_preset == "custom":
        cols = parse_str_list(actuator_columns_arg)
        if not cols:
            raise ValueError("--actuator-preset custom requires --actuator-columns")
        return cols
    if actuator_preset not in ACTUATOR_PRESETS:
        raise ValueError(f"Unknown actuator preset: {actuator_preset}")
    return list(ACTUATOR_PRESETS[actuator_preset])


def assign_split(family: str, size: float, trial_id: int) -> str:
    # Hold out only a subset of branched/irregular trials for unseen-object.
    # The rest should still be distributed across regular train/val/test.
    if family in {"branched", "irregular"} and trial_id % 5 == 0:
        return "unseen-object"
    if size >= 0.020 and trial_id % 4 == 0:
        return "unseen-geometry"
    r = trial_id % 10
    if r < 7:
        return "train"
    if r == 7:
        return "val"
    return "test"


def compute_family_class_weights_from_manifest(
    manifest_items: list[dict],
    family_labels: list[str],
    family_group_field: str,
) -> list[float]:
    train_counts = Counter()
    for item in manifest_items:
        if item["split"] == "train":
            train_counts[item[family_group_field]] += 1

    if not train_counts:
        return [1.0] * len(family_labels)

    max_count = max(train_counts.values())
    weights = []
    for label in family_labels:
        c = train_counts.get(label, 1)
        weights.append(float(max_count / max(c, 1)))
    return weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--base-xml", default=None, help="Optional. Only needed when inferring available IMUs from the private MuJoCo XML. If omitted, IMU indices are inferred from the first trial CSV.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--window-sec", type=float, default=0.40)
    ap.add_argument("--stride-sec", type=float, default=0.10)

    ap.add_argument("--imu-preset", choices=["all", "safe8", "custom"], default="all")
    ap.add_argument("--imu-indices", type=str, default=None)

    ap.add_argument(
        "--actuator-preset",
        choices=[
            "full",
            "disp_only",
            "tendon_length_only",
            "disp_and_tendon_length",
            "tendon_velocity_only",
            "custom",
        ],
        default="full",
    )
    ap.add_argument("--actuator-columns", type=str, default=None)

    ap.add_argument(
        "--family-label-mode",
        choices=["exact7", "coarse5", "coarse3"],
        default="coarse5",
    )

    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    outdir = Path(args.outdir)
    windows_dir = outdir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    if args.base_xml:
        imu_layout = parse_imu_layout(args.base_xml)
        available_imu_indices = [int(name.replace("B", "")) for name, _ in imu_layout]
    else:
        available_imu_indices = infer_available_imu_indices_from_raw(raw_dir)

    imu_indices = resolve_imu_indices(
        available=available_imu_indices,
        imu_preset=args.imu_preset,
        imu_indices_arg=args.imu_indices,
    )
    actuator_columns = resolve_actuator_columns(
        actuator_preset=args.actuator_preset,
        actuator_columns_arg=args.actuator_columns,
    )

    family_info = get_family_label_info(args.family_label_mode)
    family_label_fn = family_info["label_fn"]
    family_group_name_fn = family_info["group_name_fn"]
    family_labels = family_info["family_labels"]
    num_families = family_info["num_families"]
    family_label_map = family_info["label_maps"]

    manifest: List[dict] = []
    trial_dirs = sorted((raw_dir / "trials").glob("trial_*"))
    window_counter = 0

    for trial_path in tqdm(trial_dirs, desc="windows"):
        ts_path = trial_path / "timeseries.csv"
        meta_path = trial_path / "metadata.json"
        if not ts_path.exists() or not meta_path.exists():
            continue

        df = pd.read_csv(ts_path)
        df = add_trial_features(df)

        missing_cols = [c for c in actuator_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{trial_path} is missing actuator columns: {missing_cols}")

        for idx in imu_indices:
            needed = [
                f"imu_gyro_B{idx}_x",
                f"imu_gyro_B{idx}_y",
                f"imu_gyro_B{idx}_z",
                f"imu_acc_B{idx}_x",
                f"imu_acc_B{idx}_y",
                f"imu_acc_B{idx}_z",
            ]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                raise ValueError(f"{trial_path} is missing IMU columns for B{idx}: {missing}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        exact_family = meta["object"]["family"]
        family_group = family_group_name_fn(exact_family)
        family_id = family_label_fn(exact_family)

        size = float(meta["object"]["size"])
        success = int(meta["labels"]["success"])
        quality = float(meta["labels"]["quality"])
        trial_id = int(meta["trial_id"])
        split = assign_split(exact_family, size, trial_id)
        summary = compute_trial_summary(df)

        dt = float(df["t"].diff().fillna(0.0).replace(0.0, np.nan).median())
        if not np.isfinite(dt):
            dt = 0.005
        win = max(4, int(round(args.window_sec / dt)))
        stride = max(1, int(round(args.stride_sec / dt)))

        for start in range(0, max(1, len(df) - win + 1), stride):
            end = min(len(df), start + win)
            if end - start < win:
                continue

            wdf = df.iloc[start:end].reset_index(drop=True)

            imu = np.zeros((win, len(imu_indices), 6), dtype=np.float32)
            for j, idx in enumerate(imu_indices):
                imu[:, j, 0] = wdf[f"imu_gyro_B{idx}_x"].to_numpy(np.float32)
                imu[:, j, 1] = wdf[f"imu_gyro_B{idx}_y"].to_numpy(np.float32)
                imu[:, j, 2] = wdf[f"imu_gyro_B{idx}_z"].to_numpy(np.float32)
                imu[:, j, 3] = wdf[f"imu_acc_B{idx}_x"].to_numpy(np.float32)
                imu[:, j, 4] = wdf[f"imu_acc_B{idx}_y"].to_numpy(np.float32)
                imu[:, j, 5] = wdf[f"imu_acc_B{idx}_z"].to_numpy(np.float32)

            mask = np.ones((win, len(imu_indices)), dtype=np.float32)

            actuator = np.stack(
                [wdf[col].to_numpy(np.float32) for col in actuator_columns],
                axis=-1,
            )

            phase = window_phase_label(wdf)
            action = oracle_intervention_label(wdf, summary)

            window_path = windows_dir / f"window_{window_counter:07d}.npz"
            np.savez_compressed(
                window_path,
                imu=imu,
                mask=mask,
                actuator=actuator,
                success=np.array(success, dtype=np.float32),
                quality=np.array(quality, dtype=np.float32),
                family_id=np.array(family_id, dtype=np.int64),
                size=np.array(size, dtype=np.float32),
                action_id=np.array(ACTION_TO_ID[action], dtype=np.int64),
                phase_id=np.array(PHASE_TO_ID[phase], dtype=np.int64),
                trust_target=np.ones((len(imu_indices),), dtype=np.float32),
                corruption_type=np.array("clean"),
                corruption_strength=np.array(0.0, dtype=np.float32),
                imu_order=np.array(imu_indices, dtype=np.int64),
                actuator_columns=np.array(actuator_columns),
                t_start=np.array(float(wdf["t"].iloc[0]), dtype=np.float32),
                t_end=np.array(float(wdf["t"].iloc[-1]), dtype=np.float32),
            )

            manifest.append(
                {
                    "path": str(window_path.resolve()),
                    "split": split,
                    "trial_id": trial_id,
                    "window_id": window_counter,
                    "family": exact_family,
                    "family_group": family_group,
                    "size": size,
                    "success": success,
                    "quality": quality,
                    "phase": phase,
                    "action": action,
                }
            )
            window_counter += 1

    manifest_path = outdir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")

    stats = compute_stats_from_manifest(manifest_path, split="train")
    save_stats(stats, outdir / "norm_stats.json")

    family_class_weights = compute_family_class_weights_from_manifest(
        manifest_items=manifest,
        family_labels=family_labels,
        family_group_field="family_group",
    )

    input_config = {
        "imu_preset": args.imu_preset,
        "imu_indices": imu_indices,
        "actuator_preset": args.actuator_preset,
        "actuator_columns": actuator_columns,
        "act_dim": len(actuator_columns),
        "family_label_mode": args.family_label_mode,
        "family_labels": family_labels,
        "num_families": num_families,
        "family_class_weights": family_class_weights,
    }
    with open(outdir / "input_config.json", "w", encoding="utf-8") as f:
        json.dump(input_config, f, indent=2)

    with open(outdir / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "action_to_id": ACTION_TO_ID,
                "phase_to_id": PHASE_TO_ID,
                "imu_indices": imu_indices,
                "actuator_columns": actuator_columns,
                "family_label_mode": args.family_label_mode,
                "family_labels": family_labels,
                "family_to_id": family_label_map,
                "family_class_weights": family_class_weights,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()