#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from collections import deque
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from spirob.controller import InterventionPolicy, OpenLoopGraspController, base_pull_offset
from spirob.data import load_stats
from spirob.model import ModelConfig, SpiRobTinyModel
from spirob.object_library import sample_object, workspace_positions
from spirob.sim_env import SpiRobGraspEnv
from spirob.labels import add_trial_features, compute_trial_summary


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
    "disp_only": ["ctrl_1", "ctrl_2"],
    "tendon_length_only": ["tendon_length_1", "tendon_length_2"],
    "disp_and_tendon_length": ["ctrl_1", "ctrl_2", "tendon_length_1", "tendon_length_2"],
    "tendon_velocity_only": ["tendon_velocity_1", "tendon_velocity_2"],
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
        chosen = parse_int_list(imu_indices_arg)
        if not chosen:
            raise ValueError("--imu-preset custom requires --imu-indices")
        missing = [i for i in chosen if i not in available]
        if missing:
            raise ValueError(f"Requested IMUs not available: {missing}")
    else:
        raise ValueError(f"Unknown imu preset: {imu_preset}")
    if not chosen:
        raise ValueError("No IMUs selected.")
    return chosen


def resolve_actuator_columns(actuator_preset: str, actuator_columns_arg: str | None) -> list[str]:
    if actuator_preset == "custom":
        cols = parse_str_list(actuator_columns_arg)
        if not cols:
            raise ValueError("--actuator-preset custom requires --actuator-columns")
        return cols
    return list(ACTUATOR_PRESETS[actuator_preset])


def obs_to_feature_row(obs, phase: str) -> dict:
    row = {
        "phase": phase,
        "ctrl_1": float(obs.ctrl[0]) if len(obs.ctrl) > 0 else 0.0,
        "ctrl_2": float(obs.ctrl[1]) if len(obs.ctrl) > 1 else 0.0,
        "actuator_force_1": float(obs.actuator_force[0]) if len(obs.actuator_force) > 0 else 0.0,
        "actuator_force_2": float(obs.actuator_force[1]) if len(obs.actuator_force) > 1 else 0.0,
        "tendon_length_1": float(obs.tendon_length[0]) if len(obs.tendon_length) > 0 else 0.0,
        "tendon_length_2": float(obs.tendon_length[1]) if len(obs.tendon_length) > 1 else 0.0,
        "tendon_velocity_1": float(obs.tendon_velocity[0]) if len(obs.tendon_velocity) > 0 else 0.0,
        "tendon_velocity_2": float(obs.tendon_velocity[1]) if len(obs.tendon_velocity) > 1 else 0.0,
        "contact": int(obs.contact),
        "obj_x": float(obs.object_pos[0]),
        "obj_y": float(obs.object_pos[1]),
        "obj_z": float(obs.object_pos[2]),
        "tip_x": float(obs.tip_pos[0]),
        "tip_y": float(obs.tip_pos[1]),
        "tip_z": float(obs.tip_pos[2]),
    }
    for m in range(obs.imu_acc.shape[0]):
        row[f"imu_gyro_B{m}_x"] = float(obs.imu_gyro[m, 0])
        row[f"imu_gyro_B{m}_y"] = float(obs.imu_gyro[m, 1])
        row[f"imu_gyro_B{m}_z"] = float(obs.imu_gyro[m, 2])
        row[f"imu_acc_B{m}_x"] = float(obs.imu_acc[m, 0])
        row[f"imu_acc_B{m}_y"] = float(obs.imu_acc[m, 1])
        row[f"imu_acc_B{m}_z"] = float(obs.imu_acc[m, 2])
    return row


def window_to_tensors(rows, stats, imu_indices: list[int], actuator_columns: list[str]):
    df = pd.DataFrame(rows)
    T = len(df)
    M = len(imu_indices)
    act_dim = len(actuator_columns)

    imu = np.zeros((1, T, M, 6), dtype=np.float32)
    for j, idx in enumerate(imu_indices):
        imu[0, :, j, 0] = df[f"imu_gyro_B{idx}_x"].to_numpy(np.float32)
        imu[0, :, j, 1] = df[f"imu_gyro_B{idx}_y"].to_numpy(np.float32)
        imu[0, :, j, 2] = df[f"imu_gyro_B{idx}_z"].to_numpy(np.float32)
        imu[0, :, j, 3] = df[f"imu_acc_B{idx}_x"].to_numpy(np.float32)
        imu[0, :, j, 4] = df[f"imu_acc_B{idx}_y"].to_numpy(np.float32)
        imu[0, :, j, 5] = df[f"imu_acc_B{idx}_z"].to_numpy(np.float32)

    mask = np.ones((1, T, M), dtype=np.float32)
    actuator = np.zeros((1, T, act_dim), dtype=np.float32)
    for k, col in enumerate(actuator_columns):
        actuator[0, :, k] = df[col].to_numpy(np.float32)

    imu = (imu - stats.imu_mean.reshape(1, 1, 1, -1)) / np.maximum(stats.imu_std.reshape(1, 1, 1, -1), 1e-6)
    actuator = (actuator - stats.act_mean.reshape(1, 1, -1)) / np.maximum(stats.act_std.reshape(1, 1, -1), 1e-6)
    return torch.from_numpy(imu), torch.from_numpy(mask), torch.from_numpy(actuator)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-xml", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num-trials", type=int, default=100)
    ap.add_argument("--window-steps", type=int, default=80)
    ap.add_argument("--decision-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cpu", action="store_true")

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

    args = ap.parse_args()

    actuator_columns = resolve_actuator_columns(args.actuator_preset, args.actuator_columns)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    stats = load_stats(args.stats)
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = SpiRobTinyModel(ModelConfig(**ckpt.get("config", {}))).to(device)
    if model.config.act_dim != len(actuator_columns):
        raise ValueError(
            f"Checkpoint expects act_dim={model.config.act_dim}, "
            f"but selected actuator columns give {len(actuator_columns)}. Retrain with matching act_dim."
        )
    model.load_state_dict(ckpt["model"])
    model.eval()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    summary_rows = []
    positions = workspace_positions()

    for trial_id in tqdm(range(args.num_trials), desc="closed-loop"):
        obj = sample_object(rng, positions=positions)
        env = SpiRobGraspEnv(args.base_xml, object_spec=obj, use_viewer=False)
        obs = env.reset()

        available = list(range(int(obs.imu_acc.shape[0])))
        imu_indices = resolve_imu_indices(available, args.imu_preset, args.imu_indices)

        controller = OpenLoopGraspController()
        controller.reset(obj.pose_x, obj.pose_z)
        policy = InterventionPolicy()
        policy.reset()

        ring = deque(maxlen=args.window_steps)
        rows = []
        interventions = []

        dt_nominal = 1.0 / 200.0
        max_steps = 1200
        for step_idx in range(max_steps):
            action = None
            if len(ring) >= args.window_steps and step_idx % args.decision_every == 0:
                imu, mask, actuator = window_to_tensors(
                    list(ring),
                    stats,
                    imu_indices=imu_indices,
                    actuator_columns=actuator_columns,
                )
                with torch.no_grad():
                    out = model(imu.to(device), mask.to(device), actuator.to(device))
                np_out = {k: v.detach().cpu().numpy() for k, v in out.items()}
                action = policy.decide(np_out, obs, controller.phase)
                interventions.append(
                    {
                        "t": obs.t,
                        "action": action,
                        "success_prob": float(np_out["success_prob"][0]),
                        "quality": float(np_out["quality"][0]),
                        "confidence": float(np_out["confidence"][0]),
                        "phase": controller.phase,
                    }
                )

            ctrl = controller.update(obs, dt=dt_nominal, intervention=action)
            mocap_pos = base_pull_offset(
                env.default_mocap_pos,
                controller.phase,
                controller.phase_time,
                controller.pull_time,
            )
            env.set_mocap_pos(mocap_pos)
            obs = env.step(ctrl)

            row = obs_to_feature_row(obs, controller.phase)
            rows.append({"t": obs.t, **row})
            ring.append(row)

            if controller.phase == "done":
                break

        df = add_trial_features(pd.DataFrame(rows))
        summary = compute_trial_summary(df)
        summary_rows.append(
            {
                "trial_id": trial_id,
                "family": obj.family,
                "size": obj.size,
                "success": summary.success,
                "quality": summary.quality,
                "n_interventions": len(interventions),
            }
        )

        with open(outdir / f"trial_{trial_id:04d}_interventions.json", "w", encoding="utf-8") as f:
            json.dump(interventions, f, indent=2)
        df.to_csv(outdir / f"trial_{trial_id:04d}_timeseries.csv", index=False)
        env.close()

    pd.DataFrame(summary_rows).to_csv(outdir / "summary.csv", index=False)


if __name__ == "__main__":
    main()