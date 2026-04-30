#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_history(path: Path):
    if path.exists():
        return load_json(path)
    return None


def infer_family_labels(metrics: dict, cm: np.ndarray | None = None) -> list[str]:
    candidates = []

    labels = metrics.get("family_label_names")
    if isinstance(labels, list) and len(labels) > 0:
        candidates.append([str(x) for x in labels])

    input_config = metrics.get("input_config", {})
    labels = input_config.get("family_labels")
    if isinstance(labels, list) and len(labels) > 0:
        candidates.append([str(x) for x in labels])

    label_maps = metrics.get("label_maps", {})
    labels = label_maps.get("family_labels")
    if isinstance(labels, list) and len(labels) > 0:
        candidates.append([str(x) for x in labels])

    if cm is not None:
        n = int(cm.shape[0])

        for cand in candidates:
            if len(cand) == n:
                return cand

        if n == 7:
            return ["sphere", "cylinder", "capsule", "box", "flat", "branched", "irregular"]
        if n == 5:
            return ["round", "elongated", "square", "flat", "irregular"]
        if n == 3:
            return ["flat", "square", "round"]
        return [f"class_{i}" for i in range(n)]

    if candidates:
        return candidates[0]

    return ["family"]


def plot_classification_summary(metrics, outdir: Path):
    clean = metrics["clean"]
    robust = metrics["robust"]

    rows = [
        ("Success AUROC", clean["success"].get("auroc"), robust["success"].get("auroc")),
        ("Action Accuracy", clean["action"].get("accuracy"), robust["action"].get("accuracy")),
        ("Family Accuracy", clean["family"].get("accuracy"), robust["family"].get("accuracy")),
    ]
    df = pd.DataFrame(rows, columns=["metric", "clean", "robust"])

    plt.figure(figsize=(9, 5))
    x = range(len(df))
    width = 0.35
    plt.bar([i - width / 2 for i in x], df["clean"], width=width, label="Clean")
    plt.bar([i + width / 2 for i in x], df["robust"], width=width, label="Robust")
    plt.xticks(list(x), df["metric"], rotation=15)
    plt.ylabel("Score")
    plt.title("Classification Metrics: Clean vs Robust")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "classification_clean_vs_robust.png", dpi=200)
    plt.close()


def plot_regression_summary(metrics, outdir: Path):
    clean = metrics["clean"]
    robust = metrics["robust"]

    rows = [
        ("Quality RMSE", clean["quality"].get("rmse"), robust["quality"].get("rmse")),
        ("Size RMSE", clean["size"].get("rmse"), robust["size"].get("rmse")),
        ("Trust RMSE", clean["trust"].get("rmse"), robust["trust"].get("rmse")),
    ]
    df = pd.DataFrame(rows, columns=["metric", "clean", "robust"])

    plt.figure(figsize=(9.5, 5))
    x = range(len(df))
    width = 0.35
    plt.bar([i - width / 2 for i in x], df["clean"], width=width, label="Clean")
    plt.bar([i + width / 2 for i in x], df["robust"], width=width, label="Robust")
    plt.xticks(list(x), df["metric"], rotation=15)
    plt.ylabel("RMSE")
    plt.title("Regression Metrics: Clean vs Robust")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "regression_clean_vs_robust.png", dpi=200)
    plt.close()


def plot_accuracy_summary(metrics, outdir: Path):
    clean = metrics["clean"]
    robust = metrics["robust"]

    rows = [
        ("Success AUROC", clean["success"].get("auroc"), robust["success"].get("auroc")),
        ("Action Accuracy", clean["action"].get("accuracy"), robust["action"].get("accuracy")),
        ("Family Accuracy", clean["family"].get("accuracy"), robust["family"].get("accuracy")),
        ("Quality RMSE", clean["quality"].get("rmse"), robust["quality"].get("rmse")),
        ("Size RMSE", clean["size"].get("rmse"), robust["size"].get("rmse")),
        ("Trust RMSE", clean["trust"].get("rmse"), robust["trust"].get("rmse")),
    ]
    df = pd.DataFrame(rows, columns=["metric", "clean", "robust"])

    plt.figure(figsize=(11.5, 5.5))
    x = range(len(df))
    width = 0.35
    plt.bar([i - width / 2 for i in x], df["clean"], width=width, label="Clean")
    plt.bar([i + width / 2 for i in x], df["robust"], width=width, label="Robust")
    plt.xticks(list(x), df["metric"], rotation=20)
    plt.ylabel("Score / Error")
    plt.title("Overall Metrics: Clean vs Robust")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "accuracy_clean_vs_robust.png", dpi=200)
    plt.close()


def plot_efficiency_summary(metrics, outdir: Path):
    eff = metrics["efficiency"]
    stats = metrics["model_stats"]

    tinyml = stats.get("tinyml_export", {})
    tinyml_kb = tinyml.get("size_kb")
    rows = [
        ("Model Size (KB)", stats.get("checkpoint_size_kb"), tinyml_kb if tinyml_kb is not None else 0.0),
        ("Parameter Count", stats.get("num_params"), stats.get("num_params")),
        ("Avg Latency (ms)", eff.get("avg_latency_ms"), eff.get("avg_latency_ms")),
        ("P95 Latency (ms)", eff.get("p95_latency_ms"), eff.get("p95_latency_ms")),
        ("Throughput (samples/s)", eff.get("throughput_samples_per_sec"), eff.get("throughput_samples_per_sec")),
    ]
    df = pd.DataFrame(rows, columns=["metric", "pytorch", "tinyml_or_same"])

    plt.figure(figsize=(10, 5.5))
    x = range(len(df))
    width = 0.35
    plt.bar([i - width / 2 for i in x], df["pytorch"], width=width, label="PyTorch")
    plt.bar([i + width / 2 for i in x], df["tinyml_or_same"], width=width, label="TinyML / Same")
    plt.xticks(list(x), df["metric"], rotation=20)
    plt.ylabel("Value")
    plt.title("Efficiency Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "efficiency_summary.png", dpi=200)
    plt.close()


def plot_latency_histogram(metrics, outdir: Path):
    times = metrics["efficiency"].get("latency_samples_ms", [])
    if not times:
        return

    plt.figure(figsize=(8, 4.5))
    plt.hist(times, bins=20)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("Inference Latency Distribution")
    plt.tight_layout()
    plt.savefig(outdir / "latency_histogram.png", dpi=200)
    plt.close()


def plot_tradeoff_accuracy_vs_latency(metrics, outdir: Path):
    success_auroc = metrics["clean"]["success"].get("auroc")
    action_acc = metrics["clean"]["action"].get("accuracy")
    family_acc = metrics["clean"]["family"].get("accuracy")
    quality_rmse = metrics["clean"]["quality"].get("rmse")
    size_rmse = metrics["clean"]["size"].get("rmse")
    trust_rmse = metrics["clean"]["trust"].get("rmse")
    latency_ms = metrics["efficiency"].get("avg_latency_ms")

    labels = [
        "Success AUROC",
        "Action Acc",
        "Family Acc",
        "Quality RMSE",
        "Size RMSE",
        "Trust RMSE",
    ]
    values = [success_auroc, action_acc, family_acc, quality_rmse, size_rmse, trust_rmse]

    plt.figure(figsize=(8, 5.5))
    for label, value in zip(labels, values):
        if value is None:
            continue
        plt.scatter(latency_ms, value)
        plt.text(latency_ms, value, f" {label}", va="center")
    plt.xlabel("Average Latency (ms)")
    plt.ylabel("Accuracy / Error")
    plt.title("Tradeoff: Metrics vs Latency")
    plt.tight_layout()
    plt.savefig(outdir / "tradeoff_accuracy_vs_latency.png", dpi=200)
    plt.close()


def plot_tradeoff_accuracy_vs_size(metrics, outdir: Path):
    success_auroc = metrics["clean"]["success"].get("auroc")
    action_acc = metrics["clean"]["action"].get("accuracy")
    family_acc = metrics["clean"]["family"].get("accuracy")
    quality_rmse = metrics["clean"]["quality"].get("rmse")
    size_rmse = metrics["clean"]["size"].get("rmse")
    trust_rmse = metrics["clean"]["trust"].get("rmse")
    model_size_kb = metrics["model_stats"].get("checkpoint_size_kb")

    labels = [
        "Success AUROC",
        "Action Acc",
        "Family Acc",
        "Quality RMSE",
        "Size RMSE",
        "Trust RMSE",
    ]
    values = [success_auroc, action_acc, family_acc, quality_rmse, size_rmse, trust_rmse]

    plt.figure(figsize=(8, 5.5))
    for label, value in zip(labels, values):
        if value is None:
            continue
        plt.scatter(model_size_kb, value)
        plt.text(model_size_kb, value, f" {label}", va="center")
    plt.xlabel("Model Size (KB)")
    plt.ylabel("Accuracy / Error")
    plt.title("Tradeoff: Metrics vs Size")
    plt.tight_layout()
    plt.savefig(outdir / "tradeoff_accuracy_vs_size.png", dpi=200)
    plt.close()


def plot_training_history(history, outdir: Path):
    if history is None:
        return

    df = pd.DataFrame(history)
    if df.empty:
        return

    if "epoch" in df.columns and "train_loss" in df.columns:
        plt.figure(figsize=(7, 4))
        plt.plot(df["epoch"], df["train_loss"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Training Loss vs Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / "training_loss.png", dpi=200)
        plt.close()

    plt.figure(figsize=(8, 4.5))
    if "success/auroc" in df.columns:
        plt.plot(df["epoch"], df["success/auroc"], marker="o", label="Success AUROC")
    if "family/accuracy" in df.columns:
        plt.plot(df["epoch"], df["family/accuracy"], marker="o", label="Family Accuracy")
    if "action/accuracy" in df.columns:
        plt.plot(df["epoch"], df["action/accuracy"], marker="o", label="Action Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Classification Metrics")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "validation_classification.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    if "quality/rmse" in df.columns:
        plt.plot(df["epoch"], df["quality/rmse"], marker="o", label="Quality RMSE")
    if "size/rmse" in df.columns:
        plt.plot(df["epoch"], df["size/rmse"], marker="o", label="Size RMSE")
    if "trust/rmse" in df.columns:
        plt.plot(df["epoch"], df["trust/rmse"], marker="o", label="Trust RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Validation Regression Metrics")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "validation_regression.png", dpi=200)
    plt.close()


def plot_family_confusion_matrix(metrics, outdir: Path, mode: str = "clean"):
    block = metrics.get(mode, {})
    cm = block.get("family_confusion_matrix")
    if cm is None:
        return

    cm = np.asarray(cm, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Invalid family confusion matrix shape for {mode}: {cm.shape}")

    labels = infer_family_labels(metrics, cm=cm)

    if len(labels) != cm.shape[0]:
        raise ValueError(
            f"Label count mismatch for {mode}: "
            f"{len(labels)} labels but confusion matrix is {cm.shape[0]}x{cm.shape[1]}. "
            f"Labels={labels}"
        )

    plt.figure(figsize=(7.5, 6.2))
    plt.imshow(cm, aspect="auto")
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Family Confusion Matrix ({mode.capitalize()})")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outdir / f"family_confusion_matrix_{mode}.png", dpi=200)
    plt.close()


def _extract_family_dict(block: dict, candidate_keys: list[str]) -> dict | None:
    for key in candidate_keys:
        value = block.get(key)
        if isinstance(value, dict) and len(value) > 0:
            return value
    return None


def plot_family_metric_bars(metrics, outdir: Path, mode: str = "clean"):
    block = metrics.get(mode, {})

    family_acc = _extract_family_dict(
        block,
        [
            "family_accuracy_by_class",
            "family_acc_by_class",
            "family_metrics_by_class",
            "family_accuracy_by_family",
        ],
    )
    if family_acc is not None:
        labels = list(family_acc.keys())
        values = [float(family_acc[k]) for k in labels]

        plt.figure(figsize=(8.5, 4.8))
        plt.bar(labels, values)
        plt.ylabel("Accuracy")
        plt.xlabel("Object Family")
        plt.title(f"Family Accuracy by Object Family ({mode.capitalize()})")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / f"family_accuracy_by_family_{mode}.png", dpi=200)
        plt.close()

    family_success = _extract_family_dict(
        block,
        [
            "success_rate_by_family",
            "family_success_rate",
            "closed_loop_success_rate_by_family",
            "success_by_family",
        ],
    )
    if family_success is not None:
        labels = list(family_success.keys())
        values = [
            100.0 * float(family_success[k]) if float(family_success[k]) <= 1.0 else float(family_success[k])
            for k in labels
        ]

        plt.figure(figsize=(8.5, 4.8))
        plt.bar(labels, values)
        plt.ylabel("Success Rate (%)")
        plt.xlabel("Object Family")
        plt.title(f"Success Rate by Object Family ({mode.capitalize()})")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / f"success_rate_by_family_{mode}.png", dpi=200)
        plt.close()


def plot_family_from_confusion(metrics, outdir: Path, mode: str = "clean"):
    block = metrics.get(mode, {})
    cm = block.get("family_confusion_matrix")
    if cm is None:
        return

    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] == 0 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Invalid family confusion matrix shape for {mode}: {cm.shape}")

    labels = infer_family_labels(metrics, cm=cm)

    if len(labels) != cm.shape[0]:
        raise ValueError(
            f"Label count mismatch for {mode}: "
            f"{len(labels)} labels but confusion matrix is {cm.shape[0]}x{cm.shape[1]}. "
            f"Labels={labels}"
        )

    row_sums = cm.sum(axis=1)
    recalls = []
    for i in range(len(labels)):
        if row_sums[i] > 0:
            recalls.append(float(cm[i, i] / row_sums[i]))
        else:
            recalls.append(0.0)

    plt.figure(figsize=(8.5, 4.8))
    plt.bar(labels, recalls)
    plt.ylabel("Recall")
    plt.xlabel("Object Family")
    plt.title(f"Per-Family Recall from Confusion Matrix ({mode.capitalize()})")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / f"family_recall_from_confusion_{mode}.png", dpi=200)
    plt.close()


def save_summary_text(history, metrics, outdir: Path):
    lines = []
    lines.append("Accuracy, Efficiency, and Tradeoff Summary")
    lines.append("=" * 40)

    if history is not None:
        df = pd.DataFrame(history)
        if not df.empty:
            best_row = df.loc[df["success/auroc"].idxmax()] if "success/auroc" in df.columns else df.iloc[-1]
            lines.append(f"Best epoch: {int(best_row['epoch'])}")
            if "train_loss" in best_row:
                lines.append(f"Train loss: {best_row['train_loss']:.4f}")
            if "success/auroc" in best_row:
                lines.append(f"Val success AUROC: {best_row['success/auroc']:.4f}")
            if "family/accuracy" in best_row:
                lines.append(f"Val family accuracy: {best_row['family/accuracy']:.4f}")
            if "action/accuracy" in best_row:
                lines.append(f"Val action accuracy: {best_row['action/accuracy']:.4f}")
            if "quality/rmse" in best_row:
                lines.append(f"Val quality RMSE: {best_row['quality/rmse']:.4f}")
            if "size/rmse" in best_row:
                lines.append(f"Val size RMSE: {best_row['size/rmse']:.4f}")
            if "trust/rmse" in best_row:
                lines.append(f"Val trust RMSE: {best_row['trust/rmse']:.4f}")
            lines.append("")

    clean = metrics["clean"]
    robust = metrics["robust"]
    eff = metrics["efficiency"]
    stats = metrics["model_stats"]

    lines.append("Classification:")
    lines.append(f"  Clean success AUROC: {clean['success'].get('auroc', float('nan')):.4f}")
    lines.append(f"  Clean action accuracy: {clean['action'].get('accuracy', float('nan')):.4f}")
    lines.append(f"  Clean family accuracy: {clean['family'].get('accuracy', float('nan')):.4f}")
    lines.append(f"  Robust success AUROC: {robust['success'].get('auroc', float('nan')):.4f}")
    lines.append(f"  Robust action accuracy: {robust['action'].get('accuracy', float('nan')):.4f}")
    lines.append(f"  Robust family accuracy: {robust['family'].get('accuracy', float('nan')):.4f}")
    lines.append("")

    lines.append("Regression:")
    lines.append(f"  Clean quality RMSE: {clean['quality'].get('rmse', float('nan')):.4f}")
    lines.append(f"  Clean size RMSE: {clean['size'].get('rmse', float('nan')):.4f}")
    lines.append(f"  Clean trust RMSE: {clean['trust'].get('rmse', float('nan')):.4f}")
    lines.append(f"  Robust quality RMSE: {robust['quality'].get('rmse', float('nan')):.4f}")
    lines.append(f"  Robust size RMSE: {robust['size'].get('rmse', float('nan')):.4f}")
    lines.append(f"  Robust trust RMSE: {robust['trust'].get('rmse', float('nan')):.4f}")
    lines.append("")

    lines.append("Efficiency:")
    lines.append(f"  Model size (KB): {stats.get('checkpoint_size_kb', float('nan')):.2f}")
    lines.append(f"  Parameter count: {stats.get('num_params', 0)}")
    lines.append(f"  Average latency (ms): {eff.get('avg_latency_ms', float('nan')):.4f}")
    lines.append(f"  P95 latency (ms): {eff.get('p95_latency_ms', float('nan')):.4f}")
    lines.append(f"  Throughput (samples/s): {eff.get('throughput_samples_per_sec', float('nan')):.2f}")

    tinyml = stats.get("tinyml_export", {})
    tinyml_size_kb = tinyml.get("size_kb")
    if tinyml_size_kb is not None:
        lines.append(f"  TinyML export size (KB): {tinyml_size_kb:.2f}")
    else:
        lines.append("  TinyML export size (KB): not provided")
    lines.append("")

    lines.append("Tradeoff:")
    lines.append(
        f"  Accuracy vs latency uses clean success AUROC={clean['success'].get('auroc', float('nan')):.4f} "
        f"at avg latency={eff.get('avg_latency_ms', float('nan')):.4f} ms"
    )
    lines.append(
        f"  Accuracy vs size uses clean success AUROC={clean['success'].get('auroc', float('nan')):.4f} "
        f"at model size={stats.get('checkpoint_size_kb', float('nan')):.2f} KB"
    )

    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default="runs/model_student")
    ap.add_argument("--eval-dir", type=str, default="runs/eval")
    ap.add_argument("--outdir", type=str, default="runs/figures")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    eval_dir = Path(args.eval_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    history = maybe_load_history(model_dir / "history.json")
    metrics = load_json(eval_dir / "metrics.json")

    plot_training_history(history, outdir)

    plot_classification_summary(metrics, outdir)
    plot_regression_summary(metrics, outdir)
    plot_accuracy_summary(metrics, outdir)

    plot_efficiency_summary(metrics, outdir)
    plot_latency_histogram(metrics, outdir)
    plot_tradeoff_accuracy_vs_latency(metrics, outdir)
    plot_tradeoff_accuracy_vs_size(metrics, outdir)

    plot_family_confusion_matrix(metrics, outdir, mode="clean")
    plot_family_confusion_matrix(metrics, outdir, mode="robust")

    plot_family_metric_bars(metrics, outdir, mode="clean")
    plot_family_metric_bars(metrics, outdir, mode="robust")

    plot_family_from_confusion(metrics, outdir, mode="clean")
    plot_family_from_confusion(metrics, outdir, mode="robust")

    save_summary_text(history, metrics, outdir)

    print(f"Saved figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()