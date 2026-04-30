#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def load_metadata(trial_dir: Path) -> dict | None:
    meta_path = trial_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_success_label(meta: dict) -> int | None:
    try:
        return int(meta["labels"]["success"])
    except Exception:
        return None


def get_quality_label(meta: dict) -> float | None:
    try:
        return float(meta["labels"]["quality"])
    except Exception:
        return None


def collect_trials(raw_dir: Path) -> list[tuple[Path, dict]]:
    trials_root = raw_dir / "trials"
    if not trials_root.exists():
        raise FileNotFoundError(f"Could not find trials directory: {trials_root}")

    items: list[tuple[Path, dict]] = []
    for trial_dir in sorted(trials_root.glob("trial_*")):
        meta = load_metadata(trial_dir)
        if meta is None:
            continue
        items.append((trial_dir, meta))
    return items


def filter_trials(
    items: list[tuple[Path, dict]],
    mode: str,
    quality_min: float | None,
    quality_max: float | None,
) -> list[tuple[Path, dict]]:
    out: list[tuple[Path, dict]] = []

    for trial_dir, meta in items:
        success = get_success_label(meta)
        quality = get_quality_label(meta)

        if success is None or quality is None:
            continue

        keep = False
        if mode == "success":
            keep = success == 1
        elif mode == "failure":
            keep = success == 0
        elif mode == "all":
            keep = True
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not keep:
            continue

        if quality_min is not None and quality < quality_min:
            continue
        if quality_max is not None and quality > quality_max:
            continue

        out.append((trial_dir, meta))

    return out


def print_trial_summary(trial_dir: Path, meta: dict) -> None:
    obj = meta.get("object", {})
    labels = meta.get("labels", {})

    print("=" * 70)
    print(f"Trial folder : {trial_dir}")
    print(f"Trial id     : {meta.get('trial_id')}")
    print(f"Family       : {obj.get('family')}")
    print(f"Size         : {obj.get('size')}")
    print(f"Mass         : {obj.get('mass')}")
    print(f"Pose x/z     : ({obj.get('pose_x')}, {obj.get('pose_z')})")
    print(f"Yaw          : {obj.get('yaw')}")
    print(f"Success      : {labels.get('success')}")
    print(f"Quality      : {labels.get('quality')}")
    print(f"Pull force   : {labels.get('pull_force_proxy')}")
    print(f"Hold time    : {labels.get('hold_time')}")
    print(f"Slip penalty : {labels.get('slip_penalty')}")
    print(f"Attach ratio : {labels.get('attached_ratio_pull')}")
    print(f"Lift dist    : {labels.get('lift_distance')}")
    print(f"Metadata     : {trial_dir / 'metadata.json'}")
    print(f"Timeseries   : {trial_dir / 'timeseries.csv'}")


def copy_selected_trials(selected: list[tuple[Path, dict]], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for trial_dir, _ in selected:
        dst = outdir / trial_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(trial_dir, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="Path like runs/raw_dataset")
    ap.add_argument(
        "--mode",
        choices=["success", "failure", "all"],
        default="all",
        help="Sample successful trials, failed trials, or all",
    )
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quality-min", type=float, default=None)
    ap.add_argument("--quality-max", type=float, default=None)
    ap.add_argument(
        "--copy-to",
        type=str,
        default="",
        help="Optional output folder to copy sampled trials into",
    )
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    rng = random.Random(args.seed)

    items = collect_trials(raw_dir)
    filtered = filter_trials(
        items,
        mode=args.mode,
        quality_min=args.quality_min,
        quality_max=args.quality_max,
    )

    if not filtered:
        print("No matching trials found.")
        return

    k = min(args.num_samples, len(filtered))
    selected = rng.sample(filtered, k=k)

    print(f"Found {len(filtered)} matching trials. Showing {k}.")
    for trial_dir, meta in selected:
        print_trial_summary(trial_dir, meta)

    if args.copy_to:
        copy_selected_trials(selected, Path(args.copy_to))
        print("=" * 70)
        print(f"Copied selected trials to: {Path(args.copy_to).resolve()}")


if __name__ == "__main__":
    main()