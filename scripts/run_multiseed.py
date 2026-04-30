#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import math
import subprocess

import numpy as np
import pandas as pd



def ci95_from_seed_values(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95_halfwidth": float("nan")}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    half = 1.96 * std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95_halfwidth": float(half)}



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seeds", type=str, default="3,7,11,19,23")
    ap.add_argument("--train-extra", type=str, default="")
    ap.add_argument("--eval-extra", type=str, default="")
    ap.add_argument("--teacher", type=str, default="")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    train_extra = [x for x in args.train_extra.split() if x]
    eval_extra = [x for x in args.eval_extra.split() if x]

    rows = []
    for seed in seeds:
        seed_dir = outdir / f"seed_{seed}"
        train_dir = seed_dir / "train"
        eval_dir = seed_dir / "eval_test"
        train_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--data-dir", args.data_dir,
            "--outdir", str(train_dir),
            "--seed", str(seed),
        ]
        if args.teacher:
            train_cmd += ["--teacher", args.teacher]
        train_cmd += train_extra
        subprocess.run(train_cmd, check=True)

        eval_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate.py"),
            "--data-dir", args.data_dir,
            "--checkpoint", str(train_dir / "best.pt"),
            "--outdir", str(eval_dir),
            "--split", "test",
        ] + eval_extra
        subprocess.run(eval_cmd, check=True)

        metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": seed,
                "clean_trial_success_auroc": metrics["clean_trial"]["success"]["auroc"],
                "clean_trial_family_accuracy": metrics["clean_trial"]["family"]["accuracy"],
                "clean_trial_quality_rmse": metrics["clean_trial"]["quality"]["rmse"],
                "clean_trial_size_rmse": metrics["clean_trial"]["size"]["rmse"],
                "clean_window_action_accuracy": metrics["clean"]["action"]["accuracy"],
                "clean_window_trust_rmse": metrics["clean"]["trust"]["rmse"],
                "robust_trial_success_auroc": metrics["robust_trial"]["success"]["auroc"],
                "robust_trial_family_accuracy": metrics["robust_trial"]["family"]["accuracy"],
                "robust_trial_quality_rmse": metrics["robust_trial"]["quality"]["rmse"],
                "robust_trial_size_rmse": metrics["robust_trial"]["size"]["rmse"],
                "robust_window_action_accuracy": metrics["robust"]["action"]["accuracy"],
                "robust_window_trust_rmse": metrics["robust"]["trust"]["rmse"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "multiseed_runs.csv", index=False)

    summary = {}
    for col in df.columns:
        if col == "seed":
            continue
        summary[col] = ci95_from_seed_values(df[col].astype(float).tolist())

    with open(outdir / "multiseed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_rows = []
    for metric_name, stats in summary.items():
        summary_rows.append({"metric": metric_name, **stats})
    pd.DataFrame(summary_rows).to_csv(outdir / "multiseed_summary.csv", index=False)
    print(f"Saved multi-seed summary to: {outdir}")


if __name__ == "__main__":
    main()
