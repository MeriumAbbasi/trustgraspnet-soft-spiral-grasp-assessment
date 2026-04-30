#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from spirob.corruption import CorruptionConfig
from spirob.data import SpiRobWindowDataset, collate_fn, load_stats
from spirob.metrics import (
    binary_metrics,
    multiclass_metrics,
    regression_metrics,
    robustness_breakdown,
    trust_metrics,
)
from spirob.model import ModelConfig, SpiRobTinyModel


NUM_ACTIONS = 4



def confusion_matrix_from_preds(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> list[list[int]]:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm.tolist()



def counts_from_labels(y: np.ndarray, num_classes: int) -> list[int]:
    counts = np.zeros((num_classes,), dtype=np.int64)
    for t in y.astype(int):
        if 0 <= t < num_classes:
            counts[t] += 1
    return counts.tolist()



def resolve_family_labels(input_config: dict, expected_num_families: int | None = None) -> list[str]:
    labels = input_config.get("family_labels")
    if isinstance(labels, list) and len(labels) > 0:
        labels = [str(x) for x in labels]
        if expected_num_families is None or len(labels) == expected_num_families:
            return labels

    labels = input_config.get("coarse_shape_labels")
    if isinstance(labels, list) and len(labels) > 0:
        labels = [str(x) for x in labels]
        if expected_num_families is None or len(labels) == expected_num_families:
            return labels

    mode = input_config.get("family_label_mode")
    if mode == "exact7":
        labels = ["sphere", "cylinder", "capsule", "box", "flat", "branched", "irregular"]
        if expected_num_families is None or len(labels) == expected_num_families:
            return labels
    if mode == "coarse5":
        labels = ["round", "elongated", "square", "flat", "irregular"]
        if expected_num_families is None or len(labels) == expected_num_families:
            return labels
    if mode == "coarse3":
        labels = ["round", "elongated", "square"]
        if expected_num_families is None or len(labels) == expected_num_families:
            return labels

    if expected_num_families is not None and expected_num_families > 0:
        return [f"class_{i}" for i in range(expected_num_families)]

    num_families = int(input_config.get("num_families", 0))
    if num_families > 0:
        return [f"class_{i}" for i in range(num_families)]

    return ["family"]



def regression_breakdown(corruption_types, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    corruption_types = list(corruption_types)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = {}
    unique_corr = sorted(set(corruption_types))
    for corr in unique_corr:
        idx = np.asarray([c == corr for c in corruption_types], dtype=bool)
        if idx.sum() == 0:
            continue
        out[corr] = regression_metrics(y_true[idx], y_pred[idx])
    return out



def trust_breakdown(corruption_types, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    corruption_types = list(corruption_types)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = {}
    unique_corr = sorted(set(corruption_types))
    for corr in unique_corr:
        idx = np.asarray([c == corr for c in corruption_types], dtype=bool)
        if idx.sum() == 0:
            continue
        out[corr] = trust_metrics(y_true[idx], y_pred[idx])
    return out


@torch.no_grad()
def benchmark_latency(model, device, imu_shape, mask_shape, actuator_shape, runs: int = 100, warmup: int = 20):
    model.eval()

    imu = torch.randn(*imu_shape, device=device)
    mask = torch.ones(*mask_shape, device=device)
    actuator = torch.randn(*actuator_shape, device=device)

    for _ in range(warmup):
        _ = model(imu, mask, actuator)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(imu, mask, actuator)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    avg_ms = float(arr.mean())
    p95_ms = float(np.percentile(arr, 95))
    throughput = float(1000.0 / avg_ms) if avg_ms > 0 else float("nan")

    return {
        "avg_latency_ms": avg_ms,
        "p95_latency_ms": p95_ms,
        "throughput_samples_per_sec": throughput,
        "latency_samples_ms": arr.tolist(),
    }



def _window_report_from_df(df: pd.DataFrame, num_families: int) -> dict:
    success_y = df["success_true"].to_numpy(dtype=float)
    success_p = df["success_prob"].to_numpy(dtype=float)
    quality_y = df["quality_true"].to_numpy(dtype=float)
    quality_p = df["quality_pred"].to_numpy(dtype=float)
    family_y = df["family_true"].to_numpy(dtype=int)
    family_logit_cols = [f"family_logit_{i}" for i in range(num_families)]
    family_logits = df[family_logit_cols].to_numpy(dtype=float)
    size_y = df["size_true"].to_numpy(dtype=float)
    size_p = df["size_pred"].to_numpy(dtype=float)
    action_y = df["action_true"].to_numpy(dtype=int)
    action_logit_cols = [f"action_logit_{i}" for i in range(NUM_ACTIONS)]
    action_logits = df[action_logit_cols].to_numpy(dtype=float)
    trust_true = np.stack(df["trust_true_vec"].to_list()).astype(float)
    trust_pred = np.stack(df["trust_pred_vec"].to_list()).astype(float)
    all_corr = df["corruption_type"].astype(str).tolist()

    family_pred = np.argmax(family_logits, axis=1)

    report = {}
    report["success"] = binary_metrics(success_y, success_p)
    report["quality"] = regression_metrics(quality_y, quality_p)
    report["family"] = multiclass_metrics(family_y, family_logits)
    report["size"] = regression_metrics(size_y, size_p)
    report["action"] = multiclass_metrics(action_y, action_logits)
    report["trust"] = trust_metrics(trust_true, trust_pred)

    report["robustness"] = robustness_breakdown(all_corr, success_y, success_p)
    report["quality_robustness"] = regression_breakdown(all_corr, quality_y, quality_p)
    report["size_robustness"] = regression_breakdown(all_corr, size_y, size_p)
    report["trust_robustness"] = trust_breakdown(all_corr, trust_true, trust_pred)

    report["family_confusion_matrix"] = confusion_matrix_from_preds(
        family_y, family_pred, num_classes=num_families
    )
    report["family_true_labels"] = family_y.astype(int).tolist()
    report["family_pred_labels"] = family_pred.astype(int).tolist()
    report["family_true_counts"] = counts_from_labels(family_y, num_families)
    report["family_pred_counts"] = counts_from_labels(family_pred, num_families)
    report["n_windows"] = int(len(df))
    report["n_trials"] = int(df["trial_id"].nunique())
    return report



def _aggregate_trial_predictions(df: pd.DataFrame, num_families: int) -> pd.DataFrame:
    family_logit_cols = [f"family_logit_{i}" for i in range(num_families)]
    action_logit_cols = [f"action_logit_{i}" for i in range(NUM_ACTIONS)]
    rows = []

    for trial_id, g in df.groupby("trial_id", sort=True):
        row = {
            "trial_id": int(trial_id),
            "split": str(g["split"].iloc[0]),
            "success_true": float(g["success_true"].iloc[0]),
            "success_prob": float(g["success_prob"].mean()),
            "quality_true": float(g["quality_true"].iloc[0]),
            "quality_pred": float(g["quality_pred"].mean()),
            "family_true": int(g["family_true"].iloc[0]),
            "size_true": float(g["size_true"].iloc[0]),
            "size_pred": float(g["size_pred"].mean()),
            "num_windows": int(len(g)),
            "corruption_type": str(g["corruption_type"].mode().iloc[0]),
            "mean_window_trust_rmse": float(g["trust_rmse_window"].mean()),
        }
        fam_logits = g[family_logit_cols].mean(axis=0).to_numpy(dtype=float)
        for i, v in enumerate(fam_logits):
            row[f"family_logit_{i}"] = float(v)
        row["family_pred"] = int(np.argmax(fam_logits))

        act_logits = g[action_logit_cols].mean(axis=0).to_numpy(dtype=float)
        for i, v in enumerate(act_logits):
            row[f"action_logit_{i}"] = float(v)
        # Keep majority action only as auxiliary info; action is fundamentally a window-level task.
        row["action_true_mode"] = int(g["action_true"].mode().iloc[0])
        row["action_pred_mode"] = int(np.argmax(act_logits))

        rows.append(row)

    return pd.DataFrame(rows)



def _trial_report_from_df(trial_df: pd.DataFrame, num_families: int) -> dict:
    family_logit_cols = [f"family_logit_{i}" for i in range(num_families)]
    family_logits = trial_df[family_logit_cols].to_numpy(dtype=float)

    report = {
        "success": binary_metrics(trial_df["success_true"].to_numpy(dtype=float), trial_df["success_prob"].to_numpy(dtype=float)),
        "quality": regression_metrics(trial_df["quality_true"].to_numpy(dtype=float), trial_df["quality_pred"].to_numpy(dtype=float)),
        "family": multiclass_metrics(trial_df["family_true"].to_numpy(dtype=int), family_logits),
        "size": regression_metrics(trial_df["size_true"].to_numpy(dtype=float), trial_df["size_pred"].to_numpy(dtype=float)),
        "action_majority": {
            "accuracy": float(np.mean(trial_df["action_true_mode"].to_numpy(dtype=int) == trial_df["action_pred_mode"].to_numpy(dtype=int)))
            if len(trial_df) else 0.0
        },
        "trust": {
            "mean_window_rmse": float(trial_df["mean_window_trust_rmse"].mean()) if len(trial_df) else 0.0,
        },
        "n_trials": int(len(trial_df)),
    }
    return report



def _bootstrap_trial_ci(trial_df: pd.DataFrame, num_families: int, n_boot: int = 1000, seed: int = 123) -> dict:
    if len(trial_df) == 0:
        return {}

    family_logit_cols = [f"family_logit_{i}" for i in range(num_families)]
    rng = np.random.default_rng(seed)
    metrics = {
        "success_auroc": [],
        "success_accuracy": [],
        "family_accuracy": [],
        "quality_rmse": [],
        "size_rmse": [],
    }

    for _ in range(n_boot):
        idx = rng.integers(0, len(trial_df), size=len(trial_df))
        sample = trial_df.iloc[idx].reset_index(drop=True)
        success_m = binary_metrics(sample["success_true"].to_numpy(dtype=float), sample["success_prob"].to_numpy(dtype=float))
        family_m = multiclass_metrics(sample["family_true"].to_numpy(dtype=int), sample[family_logit_cols].to_numpy(dtype=float))
        quality_m = regression_metrics(sample["quality_true"].to_numpy(dtype=float), sample["quality_pred"].to_numpy(dtype=float))
        size_m = regression_metrics(sample["size_true"].to_numpy(dtype=float), sample["size_pred"].to_numpy(dtype=float))
        metrics["success_auroc"].append(success_m["auroc"])
        metrics["success_accuracy"].append(success_m["accuracy"])
        metrics["family_accuracy"].append(family_m["accuracy"])
        metrics["quality_rmse"].append(quality_m["rmse"])
        metrics["size_rmse"].append(size_m["rmse"])

    out = {}
    for name, values in metrics.items():
        arr = np.asarray(values, dtype=float)
        out[name] = {
            "mean": float(arr.mean()),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    return out


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    rows = []

    for batch in loader:
        imu = batch["imu"].to(device)
        mask = batch["mask"].to(device)
        actuator = batch["actuator"].to(device)
        out = model(imu, mask, actuator)

        success_prob = out["success_prob"].cpu().numpy()
        quality_pred = out["quality"].cpu().numpy()
        family_logits = out["family_logits"].cpu().numpy()
        size_pred = out["size"].cpu().numpy()
        action_logits = out["action_logits"].cpu().numpy()
        trust_pred = out["trust"].cpu().numpy()

        success_true = batch["success"].cpu().numpy()
        quality_true = batch["quality"].cpu().numpy()
        family_true = batch["family_id"].cpu().numpy()
        size_true = batch["size"].cpu().numpy()
        action_true = batch["action_id"].cpu().numpy()
        phase_id = batch["phase_id"].cpu().numpy()
        trust_true = batch["trust_target"].cpu().numpy()
        trial_id = batch["trial_id"].cpu().numpy()
        window_id = batch["window_id"].cpu().numpy()
        corruption_strength = batch["corruption_strength"].cpu().numpy()
        corruption_type = batch["corruption_type"]
        paths = batch["path"]
        splits = batch["split"]

        bs = len(success_true)
        num_families = family_logits.shape[1]
        for i in range(bs):
            row = {
                "trial_id": int(trial_id[i]),
                "window_id": int(window_id[i]),
                "split": str(splits[i]),
                "path": str(paths[i]),
                "corruption_type": str(corruption_type[i]),
                "corruption_strength": float(corruption_strength[i]),
                "success_true": float(success_true[i]),
                "success_prob": float(success_prob[i]),
                "quality_true": float(quality_true[i]),
                "quality_pred": float(quality_pred[i]),
                "family_true": int(family_true[i]),
                "family_pred": int(np.argmax(family_logits[i])),
                "size_true": float(size_true[i]),
                "size_pred": float(size_pred[i]),
                "action_true": int(action_true[i]),
                "action_pred": int(np.argmax(action_logits[i])),
                "phase_id": int(phase_id[i]),
                "trust_rmse_window": float(np.sqrt(np.mean((trust_pred[i] - trust_true[i]) ** 2))),
                "trust_true_vec": trust_true[i].astype(float).tolist(),
                "trust_pred_vec": trust_pred[i].astype(float).tolist(),
            }
            for c in range(num_families):
                row[f"family_logit_{c}"] = float(family_logits[i, c])
            for c in range(NUM_ACTIONS):
                row[f"action_logit_{c}"] = float(action_logits[i, c])
            rows.append(row)

    window_df = pd.DataFrame(rows)
    trial_df = _aggregate_trial_predictions(window_df, num_families=num_families)
    window_report = _window_report_from_df(window_df, num_families=num_families)
    trial_report = _trial_report_from_df(trial_df, num_families=num_families)
    trial_ci = _bootstrap_trial_ci(trial_df, num_families=num_families)
    return window_df, trial_df, window_report, trial_report, trial_ci



def make_loader(data_dir: Path, split: str, stats, corrupt: bool, batch_size: int):
    ds = SpiRobWindowDataset(
        data_dir / "manifest.jsonl",
        split=split,
        stats=stats,
        corrupt_train=corrupt,
        corruption_config=CorruptionConfig(p_apply=1.0),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)



def load_input_config(data_dir: Path) -> dict:
    path = data_dir / "input_config.json"
    if not path.exists():
        return {
            "act_dim": 8,
            "num_families": 4,
            "family_label_mode": "legacy_unknown",
            "family_labels": ["flat", "square", "round", "elongated"],
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def checkpoint_size_kb(path: Path) -> float:
    return float(path.stat().st_size) / 1024.0



def validate_confusion_matrix_shape(cm_list: list[list[int]], family_labels: list[str], name: str) -> None:
    cm = np.asarray(cm_list, dtype=np.int64)
    expected_shape = (len(family_labels), len(family_labels))
    if cm.shape != expected_shape:
        raise ValueError(
            f"{name} confusion matrix shape {cm.shape} does not match "
            f"{len(family_labels)} family labels: {family_labels}"
        )



def print_family_coverage(name: str, family_labels: list[str], counts: list[int]) -> None:
    print(f"\n{name} family coverage:")
    for label, count in zip(family_labels, counts):
        print(f"  {label}: {count}")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--latency-runs", type=int, default=100)
    ap.add_argument("--latency-warmup", type=int, default=20)
    ap.add_argument("--tinyml-export", type=str, default="")
    ap.add_argument("--bootstrap-samples", type=int, default=1000)
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)

    config_dict = ckpt.get("config", {})
    config = ModelConfig(**config_dict)
    model = SpiRobTinyModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    input_config = load_input_config(data_dir)

    ckpt_num_families = int(config_dict.get("num_families", config.num_families))
    family_labels = resolve_family_labels(input_config, expected_num_families=ckpt_num_families)

    if len(family_labels) != ckpt_num_families:
        raise ValueError(
            f"Family label mismatch: resolved {len(family_labels)} labels "
            f"but checkpoint expects num_families={ckpt_num_families}. "
            f"Labels={family_labels}"
        )

    stats = load_stats(data_dir / "norm_stats.json")

    clean_loader = make_loader(data_dir, args.split, stats, corrupt=False, batch_size=args.batch_size)
    clean_window_df, clean_trial_df, clean_report, clean_trial_report, clean_trial_ci = run_eval(model, clean_loader, device)
    clean_trial_ci = _bootstrap_trial_ci(clean_trial_df, num_families=ckpt_num_families, n_boot=args.bootstrap_samples)

    robust_loader = make_loader(data_dir, args.split, stats, corrupt=True, batch_size=args.batch_size)
    robust_window_df, robust_trial_df, robust_report, robust_trial_report, robust_trial_ci = run_eval(model, robust_loader, device)
    robust_trial_ci = _bootstrap_trial_ci(robust_trial_df, num_families=ckpt_num_families, n_boot=args.bootstrap_samples)

    validate_confusion_matrix_shape(clean_report["family_confusion_matrix"], family_labels, "Clean")
    validate_confusion_matrix_shape(robust_report["family_confusion_matrix"], family_labels, "Robust")

    sample_batch = next(iter(clean_loader))
    imu_shape = tuple(sample_batch["imu"][:1].to(device).shape)
    mask_shape = tuple(sample_batch["mask"][:1].to(device).shape)
    actuator_shape = tuple(sample_batch["actuator"][:1].to(device).shape)

    eff = benchmark_latency(
        model,
        device=device,
        imu_shape=imu_shape,
        mask_shape=mask_shape,
        actuator_shape=actuator_shape,
        runs=args.latency_runs,
        warmup=args.latency_warmup,
    )

    model_stats = {
        "checkpoint_size_kb": checkpoint_size_kb(ckpt_path),
        "num_params": int(sum(p.numel() for p in model.parameters())),
        "tinyml_export": {
            "path": args.tinyml_export if args.tinyml_export else None,
            "size_kb": None,
        },
    }

    if args.tinyml_export:
        tiny_path = Path(args.tinyml_export)
        if tiny_path.exists():
            model_stats["tinyml_export"]["size_kb"] = float(tiny_path.stat().st_size) / 1024.0

    report = {
        "clean": clean_report,
        "clean_trial": clean_trial_report,
        "clean_trial_ci": clean_trial_ci,
        "robust": robust_report,
        "robust_trial": robust_trial_report,
        "robust_trial_ci": robust_trial_ci,
        "efficiency": eff,
        "model_stats": model_stats,
        "family_label_names": family_labels,
        "input_config": input_config,
        "checkpoint_config": config_dict,
        "split": args.split,
    }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(outdir / "family_confusion_matrix_clean.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": family_labels,
                "matrix": clean_report["family_confusion_matrix"],
                "true_counts": clean_report["family_true_counts"],
                "pred_counts": clean_report["family_pred_counts"],
            },
            f,
            indent=2,
        )

    with open(outdir / "family_confusion_matrix_robust.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": family_labels,
                "matrix": robust_report["family_confusion_matrix"],
                "true_counts": robust_report["family_true_counts"],
                "pred_counts": robust_report["family_pred_counts"],
            },
            f,
            indent=2,
        )

    clean_window_df.to_csv(outdir / "predictions_clean_windows.csv", index=False)
    robust_window_df.to_csv(outdir / "predictions_robust_windows.csv", index=False)
    clean_trial_df.to_csv(outdir / "predictions_clean_trials.csv", index=False)
    robust_trial_df.to_csv(outdir / "predictions_robust_trials.csv", index=False)

    print(f"Saved metrics to: {outdir / 'metrics.json'}")
    print(f"Resolved family labels: {family_labels}")
    print_family_coverage("Clean true-label", family_labels, clean_report["family_true_counts"])
    print_family_coverage("Robust true-label", family_labels, robust_report["family_true_counts"])
    print("Saved trial-level CSVs for confidence-interval aggregation.")


if __name__ == "__main__":
    main()
