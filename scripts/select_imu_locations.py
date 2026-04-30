#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_SAFE8 = [0, 3, 7, 10, 12, 16, 18, 20]


def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def candidate_pool(n_segments: int, candidate_indices: str | None, forbidden_indices: str | None) -> list[int]:
    if candidate_indices:
        cand = parse_int_list(candidate_indices)
    else:
        cand = list(range(n_segments))
    forbidden = set(parse_int_list(forbidden_indices))
    return [i for i in cand if i not in forbidden]


def valid_with_spacing(selected: list[int], candidate: int, min_spacing: int) -> bool:
    return all(abs(candidate - s) >= min_spacing for s in selected)


def logdet_score(phi: np.ndarray, selected: list[int], noise_var: float, reg: float) -> float:
    H = phi[selected, :]
    F = (H.T @ H) / max(noise_var, 1e-12)
    F = F + reg * np.eye(F.shape[0], dtype=np.float64)
    sign, logdet = np.linalg.slogdet(F)
    if sign <= 0:
        return -1e18
    return float(logdet)


def greedy_d_opt(phi: np.ndarray, k: int, candidates: list[int], min_spacing: int, noise_var: float, reg: float) -> dict:
    selected = []
    progression = []
    remaining = list(candidates)

    for step in range(k):
        best_i = None
        best_score = -1e18
        for i in remaining:
            if not valid_with_spacing(selected, i, min_spacing):
                continue
            trial = selected + [i]
            score = logdet_score(phi, trial, noise_var=noise_var, reg=reg)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        progression.append({"step": step + 1, "added": int(best_i), "logdet": float(best_score)})

    return {"selected": selected, "progression": progression}


def uniform_baseline(candidates: list[int], k: int, min_spacing: int) -> list[int]:
    if len(candidates) <= k:
        return candidates[:]
    idxs = np.linspace(0, len(candidates) - 1, k).round().astype(int)
    proposal = [candidates[i] for i in idxs]
    selected = []
    for c in proposal:
        if valid_with_spacing(selected, c, min_spacing):
            selected.append(c)
    for c in candidates:
        if len(selected) >= k:
            break
        if c not in selected and valid_with_spacing(selected, c, min_spacing):
            selected.append(c)
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num-sensors", type=int, default=8)
    ap.add_argument("--candidate-indices", type=str, default=None)
    ap.add_argument("--forbidden-indices", type=str, default=None)
    ap.add_argument("--min-spacing", type=int, default=2)
    ap.add_argument("--noise-var", type=float, default=1.0)
    ap.add_argument("--reg", type=float, default=1e-6)
    args = ap.parse_args()

    modes_dir = Path(args.modes_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    phi = np.load(modes_dir / "modes.npy").astype(np.float64)
    n_segments, r = phi.shape

    candidates = candidate_pool(n_segments, args.candidate_indices, args.forbidden_indices)
    if len(candidates) < args.num_sensors:
        raise ValueError(f"Only {len(candidates)} candidates remain, but num-sensors={args.num_sensors}")

    dopt = greedy_d_opt(phi, args.num_sensors, candidates, args.min_spacing, args.noise_var, args.reg)
    uniform = uniform_baseline(candidates, args.num_sensors, args.min_spacing)
    safe8 = [i for i in DEFAULT_SAFE8 if i in candidates]

    comparison = {
        "d_opt_logdet": logdet_score(phi, dopt["selected"], args.noise_var, args.reg),
        "uniform_logdet": logdet_score(phi, uniform, args.noise_var, args.reg),
        "safe8_logdet": logdet_score(phi, safe8, args.noise_var, args.reg) if len(safe8) == args.num_sensors else None,
    }

    result = {
        "num_segments": int(n_segments),
        "num_modes": int(r),
        "num_sensors": int(args.num_sensors),
        "candidate_indices": candidates,
        "forbidden_indices": parse_int_list(args.forbidden_indices),
        "min_spacing": int(args.min_spacing),
        "noise_var": float(args.noise_var),
        "regularization": float(args.reg),
        "d_optimal": dopt,
        "uniform_baseline": uniform,
        "safe8_baseline": safe8,
        "comparison": comparison,
    }

    with open(outdir / "selected_imus.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with open(outdir / "selected_imus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join([
            "D-optimal IMU placement",
            "=" * 28,
            f"Selected IMUs: {dopt['selected']}",
            f"logdet score: {comparison['d_opt_logdet']:.6f}",
            f"Uniform baseline: {uniform}",
            f"Uniform logdet: {comparison['uniform_logdet']:.6f}",
            f"safe8 baseline: {safe8}",
            f"safe8 logdet: {comparison['safe8_logdet']}" if comparison['safe8_logdet'] is not None else "safe8 logdet: n/a",
        ]))

    print(f"Saved placement to: {outdir.resolve()}")
    print(f"D-optimal selected IMUs: {dopt['selected']}")


if __name__ == "__main__":
    main()
