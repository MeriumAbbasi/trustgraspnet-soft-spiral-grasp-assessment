#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def reconstruct_from_subset(q: np.ndarray, q_mean: np.ndarray, phi: np.ndarray, selected: list[int], reg: float) -> np.ndarray:
    y = q[selected] - q_mean[selected]
    H = phi[selected, :]
    A = H.T @ H + reg * np.eye(H.shape[1], dtype=np.float64)
    b = H.T @ y
    a_hat = np.linalg.solve(A, b)
    return q_mean + phi @ a_hat


def metrics(Q: np.ndarray, q_mean: np.ndarray, phi: np.ndarray, selected: list[int], reg: float) -> dict:
    errs = []
    tip_errs = []
    for q in Q:
        q_hat = reconstruct_from_subset(q, q_mean, phi, selected, reg)
        errs.append(np.sqrt(np.mean((q - q_hat) ** 2)))
        tip_errs.append(abs(q[-1] - q_hat[-1]))
    errs = np.asarray(errs, dtype=np.float64)
    tip_errs = np.asarray(tip_errs, dtype=np.float64)
    return {
        "shape_rmse_mean": float(errs.mean()),
        "shape_rmse_std": float(errs.std()),
        "tip_angle_error_mean": float(tip_errs.mean()),
        "tip_angle_error_std": float(tip_errs.std()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes-dir", required=True)
    ap.add_argument("--selected-indices", type=str, default=None)
    ap.add_argument("--selected-json", type=str, default=None)
    ap.add_argument("--outpath", required=True)
    ap.add_argument("--reg", type=float, default=1e-6)
    args = ap.parse_args()

    modes_dir = Path(args.modes_dir)
    outpath = Path(args.outpath)

    Q_val = np.load(modes_dir / "val_shapes.npy").astype(np.float64)
    q_mean = np.load(modes_dir / "mean_shape.npy").astype(np.float64)
    phi = np.load(modes_dir / "modes.npy").astype(np.float64)

    if args.selected_json:
        with open(args.selected_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        results = {"d_optimal": metrics(Q_val, q_mean, phi, obj["d_optimal"]["selected"], args.reg)}
        if obj.get("uniform_baseline"):
            results["uniform"] = metrics(Q_val, q_mean, phi, obj["uniform_baseline"], args.reg)
        if obj.get("safe8_baseline"):
            results["safe8"] = metrics(Q_val, q_mean, phi, obj["safe8_baseline"], args.reg)
    else:
        selected = parse_int_list(args.selected_indices)
        if not selected:
            raise ValueError("Provide either --selected-json or --selected-indices")
        results = {"selected": metrics(Q_val, q_mean, phi, selected, args.reg)}

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved reconstruction results to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
