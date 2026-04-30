#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def body_name_to_index(name: str) -> int:
    """
    Extract numeric index from names like:
    B0, B1, body_0, link12, seg_7, etc.
    """
    m = re.search(r"(\d+)", str(name))
    if not m:
        raise ValueError(f"Could not extract numeric index from body name: {name!r}")
    return int(m.group(1))


def build_theta_from_long_backbone_df(df: pd.DataFrame, max_segments: int | None = None) -> np.ndarray:
    """
    backbone.csv expected columns:
        t, body, x, y, z

    Returns:
        theta: [T, n_segments]
    """
    required = {"t", "body", "x", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"backbone.csv missing required columns: {sorted(missing)}")

    df = df.copy()
    df["body_idx"] = df["body"].map(body_name_to_index)

    times = np.sort(df["t"].unique())
    theta_rows = []

    for tval in times:
        sdf = df[df["t"] == tval].sort_values("body_idx")

        xs = sdf["x"].to_numpy(np.float64)
        zs = sdf["z"].to_numpy(np.float64)

        if len(xs) < 2:
            continue

        if max_segments is not None:
            xs = xs[: max_segments + 1]
            zs = zs[: max_segments + 1]

        if len(xs) < 2:
            continue

        dx = xs[1:] - xs[:-1]
        dz = zs[1:] - zs[:-1]
        theta = np.arctan2(dz, dx)
        theta_rows.append(theta)

    if not theta_rows:
        raise ValueError("No valid backbone timesteps found in backbone.csv")

    theta_arr = np.asarray(theta_rows, dtype=np.float64)
    theta_arr = np.unwrap(theta_arr, axis=0)
    return theta_arr


def load_trial_shapes(trial_dir: Path, max_segments: int | None) -> np.ndarray:
    backbone_path = trial_dir / "backbone.csv"
    if not backbone_path.exists():
        raise FileNotFoundError(f"Missing {backbone_path}")
    df = pd.read_csv(backbone_path)
    return build_theta_from_long_backbone_df(df, max_segments=max_segments)


def train_val_split_rows(Q: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(val_fraction * n)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        train_idx = idx[:-1]
        val_idx = idx[-1:]
    return Q[train_idx], Q[val_idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num-modes", type=int, default=6)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-segments", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=1)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trial_dirs = sorted((raw_dir / "trials").glob("trial_*"))
    if not trial_dirs:
        raise ValueError(f"No trial_* folders found under {(raw_dir / 'trials')}")

    all_shapes = []
    per_trial_counts = {}

    for trial_dir in tqdm(trial_dirs, desc="loading backbone shapes"):
        theta = load_trial_shapes(trial_dir, max_segments=args.max_segments)
        if args.subsample > 1:
            theta = theta[:: args.subsample]
        if theta.size == 0:
            continue
        all_shapes.append(theta)
        per_trial_counts[trial_dir.name] = int(theta.shape[0])

    if not all_shapes:
        raise ValueError("No usable backbone data found.")

    Q = np.concatenate(all_shapes, axis=0)  # [N, n_segments]
    q_mean = Q.mean(axis=0, keepdims=True)
    Qc = Q - q_mean

    U, S, Vt = np.linalg.svd(Qc, full_matrices=False)
    r = min(args.num_modes, Vt.shape[0])
    Phi = Vt[:r].T  # [n_segments, r]

    var = (S ** 2) / max(Q.shape[0] - 1, 1)
    explained = var / max(var.sum(), 1e-12)
    explained_r = explained[:r]

    Q_train, Q_val = train_val_split_rows(Q, args.val_fraction, args.seed)

    np.save(outdir / "all_shapes.npy", Q.astype(np.float32))
    np.save(outdir / "train_shapes.npy", Q_train.astype(np.float32))
    np.save(outdir / "val_shapes.npy", Q_val.astype(np.float32))
    np.save(outdir / "mean_shape.npy", q_mean.squeeze(0).astype(np.float32))
    np.save(outdir / "modes.npy", Phi.astype(np.float32))
    np.save(outdir / "singular_values.npy", S.astype(np.float32))
    np.save(outdir / "explained_variance_ratio.npy", explained.astype(np.float32))

    meta = {
        "raw_dir": str(raw_dir.resolve()),
        "num_trials": len(per_trial_counts),
        "num_samples": int(Q.shape[0]),
        "num_segments": int(Q.shape[1]),
        "num_modes": int(r),
        "explained_variance_ratio_first_r": explained_r.tolist(),
        "per_trial_counts": per_trial_counts,
    }
    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved shape modes to: {outdir.resolve()}")
    print(f"Samples: {Q.shape[0]}, segments: {Q.shape[1]}, modes kept: {r}")
    print(f"Explained variance (first {r}): {explained_r}")


if __name__ == "__main__":
    main()