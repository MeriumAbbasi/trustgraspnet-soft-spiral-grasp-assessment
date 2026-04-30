#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from spirob.object_library import sample_object, workspace_positions
from spirob.sim_env import SpiRobGraspEnv
from spirob.controller import OpenLoopGraspController, base_pull_offset
from spirob.labels import compute_trial_summary


def trial_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-xml", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num-trials", type=int, default=200)
    ap.add_argument("--trial-seconds", type=float, default=6.0)
    ap.add_argument("--hz", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    trials_dir = outdir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    summary_rows = []
    positions = workspace_positions()

    for trial_id in tqdm(range(args.num_trials), desc="collect"):
        obj = sample_object(rng, positions=positions)
        env = SpiRobGraspEnv(args.base_xml, object_spec=obj, use_viewer=args.viewer)
        obs = env.reset()

        controller = OpenLoopGraspController()
        controller.reset(obj.pose_x, obj.pose_z)

        trial_path = trials_dir / f"trial_{trial_id:04d}"
        trial_path.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []
        backbone_rows: list[dict] = []
        dt_nominal = 1.0 / args.hz
        steps = int(args.trial_seconds * args.hz)

        for _ in range(steps):
            ctrl = controller.update(obs, dt=dt_nominal, intervention=None)
            mocap_pos = base_pull_offset(env.default_mocap_pos, controller.phase, controller.phase_time, controller.pull_time)
            env.set_mocap_pos(mocap_pos)
            obs = env.step(ctrl, n_substeps=max(1, int(round(dt_nominal / max(env.dt, 1e-6)))))

            row = {
                "t": obs.t,
                "phase": controller.phase,
                "ctrl_1": float(obs.ctrl[0]) if len(obs.ctrl) > 0 else 0.0,
                "ctrl_2": float(obs.ctrl[1]) if len(obs.ctrl) > 1 else 0.0,
                "actuator_force_1": float(obs.actuator_force[0]) if len(obs.actuator_force) > 0 else 0.0,
                "actuator_force_2": float(obs.actuator_force[1]) if len(obs.actuator_force) > 1 else 0.0,
                "tendon_length_1": float(obs.tendon_length[0]) if len(obs.tendon_length) > 0 else 0.0,
                "tendon_length_2": float(obs.tendon_length[1]) if len(obs.tendon_length) > 1 else 0.0,
                "tendon_velocity_1": float(obs.tendon_velocity[0]) if len(obs.tendon_velocity) > 0 else 0.0,
                "tendon_velocity_2": float(obs.tendon_velocity[1]) if len(obs.tendon_velocity) > 1 else 0.0,
                "contact": int(obs.contact),
                "contact_count": int(obs.contact_count),
                "contact_force_proxy": float(obs.contact_force_proxy),
                "obj_x": float(obs.object_pos[0]),
                "obj_y": float(obs.object_pos[1]),
                "obj_z": float(obs.object_pos[2]),
                "obj_vx": float(obs.object_vel[0]),
                "obj_vy": float(obs.object_vel[1]),
                "obj_vz": float(obs.object_vel[2]),
                "tip_x": float(obs.tip_pos[0]),
                "tip_y": float(obs.tip_pos[1]),
                "tip_z": float(obs.tip_pos[2]),
            }

            for m in range(obs.imu_acc.shape[0]):
                row[f"imu_acc_B{m}_x"] = float(obs.imu_acc[m, 0])
                row[f"imu_acc_B{m}_y"] = float(obs.imu_acc[m, 1])
                row[f"imu_acc_B{m}_z"] = float(obs.imu_acc[m, 2])
                row[f"imu_gyro_B{m}_x"] = float(obs.imu_gyro[m, 0])
                row[f"imu_gyro_B{m}_y"] = float(obs.imu_gyro[m, 1])
                row[f"imu_gyro_B{m}_z"] = float(obs.imu_gyro[m, 2])
                if obs.imu_quat.shape[0] > m:
                    row[f"gt_att_B{m}_w"] = float(obs.imu_quat[m, 0])
                    row[f"gt_att_B{m}_x"] = float(obs.imu_quat[m, 1])
                    row[f"gt_att_B{m}_y"] = float(obs.imu_quat[m, 2])
                    row[f"gt_att_B{m}_z"] = float(obs.imu_quat[m, 3])

            rows.append(row)

            for b_idx, xyz in enumerate(obs.body_positions):
                backbone_rows.append({
                    "t": obs.t,
                    "body": f"B{b_idx}",
                    "x": float(xyz[0]),
                    "y": float(xyz[1]),
                    "z": float(xyz[2]),
                })

            if controller.phase == "done":
                break

        df = trial_to_dataframe(rows)
        df.to_csv(trial_path / "timeseries.csv", index=False)
        pd.DataFrame(backbone_rows).to_csv(trial_path / "backbone.csv", index=False)

        summary = compute_trial_summary(df)
        meta = {
            "trial_id": trial_id,
            "object": {
                "family": obj.family,
                "size": obj.size,
                "mass": obj.mass,
                "representative_diameter": obj.representative_diameter,
                "pose_x": obj.pose_x,
                "pose_z": obj.pose_z,
                "yaw": obj.yaw,
                "extras": obj.extras,
            },
            "labels": {
                "success": summary.success,
                "quality": summary.quality,
                "pull_force_proxy": summary.pull_force_proxy,
                "hold_time": summary.hold_time,
                "slip_penalty": summary.slip_penalty,
                "attached_ratio_pull": summary.attached_ratio_pull,
                "lift_distance": summary.lift_distance,
            },
        }
        with open(trial_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        summary_rows.append({
            "trial_id": trial_id,
            "family": obj.family,
            "size": obj.size,
            "success": summary.success,
            "quality": summary.quality,
            "pull_force_proxy": summary.pull_force_proxy,
            "hold_time": summary.hold_time,
            "slip_penalty": summary.slip_penalty,
            "attached_ratio_pull": summary.attached_ratio_pull,
        })

        env.close()

    pd.DataFrame(summary_rows).to_csv(outdir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
