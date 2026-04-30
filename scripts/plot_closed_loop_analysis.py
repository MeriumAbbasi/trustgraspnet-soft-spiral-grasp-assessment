#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def make_size_bins(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="object")
    try:
        bins = pd.qcut(df["size"], q=3, labels=["small", "medium", "large"], duplicates="drop")
        return bins.astype(str)
    except Exception:
        return pd.Series(["all"] * len(df), index=df.index, dtype="object")


def plot_success_by_family(df: pd.DataFrame, outdir: Path) -> None:
    fam = (
        df.groupby("family", as_index=False)
        .agg(
            num_trials=("trial_id", "count"),
            num_success=("success", "sum"),
            success_rate=("success", "mean"),
            mean_quality=("quality", "mean"),
        )
        .sort_values("success_rate", ascending=False)
    )
    fam["success_rate_percent"] = 100.0 * fam["success_rate"]
    fam.to_csv(outdir / "success_by_family.csv", index=False)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(fam["family"], fam["success_rate_percent"])
    plt.xlabel("Family")
    plt.ylabel("Success Rate (%)")
    plt.title("Closed-Loop Success by Family")
    plt.xticks(rotation=20)
    for bar, n in zip(bars, fam["num_trials"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(outdir / "success_by_family.png", dpi=220)
    plt.close()


def plot_success_by_size(df: pd.DataFrame, outdir: Path) -> None:
    df = df.copy()
    df["size_bin"] = make_size_bins(df)

    size_tbl = (
        df.groupby("size_bin", as_index=False)
        .agg(
            num_trials=("trial_id", "count"),
            num_success=("success", "sum"),
            success_rate=("success", "mean"),
            mean_size=("size", "mean"),
            mean_quality=("quality", "mean"),
        )
    )
    size_tbl["success_rate_percent"] = 100.0 * size_tbl["success_rate"]
    size_tbl.to_csv(outdir / "success_by_size_bin.csv", index=False)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(size_tbl["size_bin"], size_tbl["success_rate_percent"])
    plt.xlabel("Size Bin")
    plt.ylabel("Success Rate (%)")
    plt.title("Closed-Loop Success by Size")
    for bar, n in zip(bars, size_tbl["num_trials"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(outdir / "success_by_size_bin.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.5, 5))
    jitter = np.random.default_rng(0).normal(0.0, 0.01, size=len(df))
    plt.scatter(df["size"], df["success"] + jitter, alpha=0.55)
    plt.xlabel("Object Size")
    plt.ylabel("Success (jittered)")
    plt.title("Success vs Object Size")
    plt.tight_layout()
    plt.savefig(outdir / "success_vs_size_scatter.png", dpi=220)
    plt.close()


def plot_success_by_location(df: pd.DataFrame, outdir: Path) -> None:
    loc = (
        df.groupby(["pose_x", "pose_z"], as_index=False)
        .agg(
            num_trials=("trial_id", "count"),
            num_success=("success", "sum"),
            success_rate=("success", "mean"),
            mean_quality=("quality", "mean"),
        )
        .sort_values(["pose_z", "pose_x"])
    )
    loc["success_rate_percent"] = 100.0 * loc["success_rate"]
    loc.to_csv(outdir / "success_by_location.csv", index=False)

    succ = df[df["success"] == 1]
    fail = df[df["success"] == 0]

    plt.figure(figsize=(7, 6))
    if not succ.empty:
        plt.scatter(succ["pose_x"], succ["pose_z"], marker="o", alpha=0.8, label="Success")
    if not fail.empty:
        plt.scatter(fail["pose_x"], fail["pose_z"], marker="x", alpha=0.8, label="Failure")
    plt.xlabel("Workspace X")
    plt.ylabel("Workspace Z")
    plt.title("Workspace Map of Success / Failure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "workspace_success_failure_map.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        loc["pose_x"],
        loc["pose_z"],
        c=loc["success_rate_percent"],
        s=60 + 10 * loc["num_trials"],
        alpha=0.95,
    )
    plt.xlabel("Workspace X")
    plt.ylabel("Workspace Z")
    plt.title("Success Rate by Workspace Location")
    cbar = plt.colorbar(sc)
    cbar.set_label("Success Rate (%)")
    plt.tight_layout()
    plt.savefig(outdir / "success_rate_by_location.png", dpi=220)
    plt.close()


def plot_family_size_success_3d(df: pd.DataFrame, outdir: Path) -> None:
    df = df.copy()

    families = sorted(df["family"].dropna().unique().tolist())
    family_to_idx = {fam: i for i, fam in enumerate(families)}
    df["family_idx"] = df["family"].map(family_to_idx)

    # Trial-level 3D scatter
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    success_df = df[df["success"] == 1]
    fail_df = df[df["success"] == 0]

    if not success_df.empty:
        ax.scatter(
            success_df["family_idx"],
            success_df["size"],
            success_df["success"],
            marker="o",
            s=45,
            alpha=0.8,
            label="Success",
        )

    if not fail_df.empty:
        ax.scatter(
            fail_df["family_idx"],
            fail_df["size"],
            fail_df["success"],
            marker="x",
            s=45,
            alpha=0.8,
            label="Failure",
        )

    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=20, ha="right")
    ax.set_xlabel("Family")
    ax.set_ylabel("Size")
    ax.set_zlabel("Success")
    ax.set_title("3D Trial-Level Plot: Family vs Size vs Success")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "family_size_success_3d_scatter.png", dpi=220)
    plt.close()

    # Aggregated 3D plot by family
    fam_tbl = (
        df.groupby("family", as_index=False)
        .agg(
            num_trials=("trial_id", "count"),
            mean_size=("size", "mean"),
            success_rate=("success", "mean"),
            mean_quality=("quality", "mean"),
        )
        .sort_values("family")
    )
    fam_tbl["family_idx"] = fam_tbl["family"].map(family_to_idx)
    fam_tbl["success_rate_percent"] = 100.0 * fam_tbl["success_rate"]
    fam_tbl.to_csv(outdir / "family_size_success_3d_table.csv", index=False)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        fam_tbl["family_idx"],
        fam_tbl["mean_size"],
        fam_tbl["success_rate_percent"],
        s=120,
        alpha=0.9,
    )

    for _, row in fam_tbl.iterrows():
        ax.text(
            row["family_idx"],
            row["mean_size"],
            row["success_rate_percent"] + 1.0,
            f"{row['family']}\n{row['success_rate_percent']:.1f}%",
            fontsize=8,
            ha="center",
        )

    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=20, ha="right")
    ax.set_xlabel("Family")
    ax.set_ylabel("Mean Size")
    ax.set_zlabel("Success Rate (%)")
    ax.set_title("3D Family-Level Plot: Family vs Mean Size vs Success Rate")
    plt.tight_layout()
    plt.savefig(outdir / "family_size_success_3d_aggregated.png", dpi=220)
    plt.close()


def save_summary_table(df: pd.DataFrame, outdir: Path) -> None:
    overall = {
        "num_trials": int(len(df)),
        "num_success": int(df["success"].sum()),
        "num_failure": int((df["success"] == 0).sum()),
        "success_rate_percent": float(100.0 * df["success"].mean()),
        "mean_quality": float(df["quality"].mean()),
        "mean_hold_time": float(df["hold_time"].mean()),
        "mean_slip_penalty": float(df["slip_penalty"].mean()),
        "mean_attach_ratio": float(df["attached_ratio_pull"].mean()),
        "mean_num_decisions": float(df["num_decisions"].mean()),
    }
    pd.DataFrame([overall]).to_csv(outdir / "closed_loop_overall_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    summary_csv = Path(args.summary_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError(f"Summary file is empty: {summary_csv}")

    save_summary_table(df, outdir)
    plot_success_by_family(df, outdir)
    plot_success_by_size(df, outdir)
    plot_success_by_location(df, outdir)
    plot_family_size_success_3d(df, outdir)

    print(f"Saved plots and tables to: {outdir}")


if __name__ == "__main__":
    main()