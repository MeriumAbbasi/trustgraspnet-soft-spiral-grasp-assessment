#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from spirob.object_library import ObjectSpec
from spirob.sim_env import SpiRobGraspEnv
from spirob.controller import base_pull_offset


def load_meta(meta_path: Path) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_object_spec(meta: dict) -> ObjectSpec:
    obj = meta["object"]
    return ObjectSpec(
        family=obj["family"],
        size=float(obj["size"]),
        mass=float(obj["mass"]),
        representative_diameter=float(obj["representative_diameter"]),
        pose_x=float(obj["pose_x"]),
        pose_z=float(obj["pose_z"]),
        yaw=float(obj["yaw"]),
        extras=obj.get("extras", {}),
    )


def compute_phase_time_column(df: pd.DataFrame) -> pd.Series:
    if "phase" not in df.columns or "t" not in df.columns:
        return pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)

    phase_time = np.zeros(len(df), dtype=np.float32)
    current_phase = None
    phase_start_t = 0.0

    for i, row in df.iterrows():
        phase = str(row["phase"])
        t = float(row["t"])

        if phase != current_phase:
            current_phase = phase
            phase_start_t = t

        phase_time[i] = t - phase_start_t

    return pd.Series(phase_time, index=df.index)


def print_trial_info(trial_dir: Path, meta: dict) -> None:
    obj = meta.get("object", {})
    labels = meta.get("labels", {})

    print("=" * 80)
    print(f"Trial folder         : {trial_dir}")
    print(f"Trial id             : {meta.get('trial_id')}")
    print(f"Family               : {obj.get('family')}")
    print(f"Size                 : {obj.get('size')}")
    print(f"Mass                 : {obj.get('mass')}")
    print(f"Pose x/z             : ({obj.get('pose_x')}, {obj.get('pose_z')})")
    print(f"Yaw                  : {obj.get('yaw')}")
    print("-" * 80)
    print(f"SUCCESS LABEL        : {labels.get('success')}")
    print(f"QUALITY LABEL        : {labels.get('quality')}")
    print(f"PULL FORCE PROXY     : {labels.get('pull_force_proxy')}")
    print(f"HOLD TIME            : {labels.get('hold_time')}")
    print(f"SLIP PENALTY         : {labels.get('slip_penalty')}")
    print(f"ATTACHED RATIO PULL  : {labels.get('attached_ratio_pull')}")
    print(f"LIFT DISTANCE        : {labels.get('lift_distance')}")
    print("=" * 80)


def try_set_fixed_camera(env) -> None:
    try:
        import mujoco

        if hasattr(env, "viewer") and env.viewer is not None:
            cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "camera1")
            if cam_id >= 0:
                env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                env.viewer.cam.fixedcamid = int(cam_id)
                if hasattr(env.viewer.cam, "trackbodyid"):
                    env.viewer.cam.trackbodyid = -1
                print(f"Using fixed XML camera: camera1 (id={cam_id})")
            else:
                print("camera1 not found in XML; using default viewer camera.")
    except Exception as e:
        print(f"Could not set fixed XML camera: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-dir", required=True, help="e.g. runs/raw_dataset/trials/trial_0369")
    ap.add_argument("--base-xml", required=True, help="e.g. xml/2Dsimreal_grasp.xml")
    ap.add_argument("--viewer-sleep", type=float, default=0.03)
    ap.add_argument("--use-mocap-replay", action="store_true")
    ap.add_argument("--pull-time", type=float, default=1.20)
    ap.add_argument("--fixed-camera", action="store_true")
    ap.add_argument("--auto-exit", action="store_true")
    args = ap.parse_args()

    trial_dir = Path(args.trial_dir)
    meta_path = trial_dir / "metadata.json"
    ts_path = trial_dir / "timeseries.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {meta_path}")
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing timeseries.csv: {ts_path}")

    meta = load_meta(meta_path)
    obj_spec = load_object_spec(meta)
    df = pd.read_csv(ts_path)

    if "ctrl_1" not in df.columns or "ctrl_2" not in df.columns:
        raise ValueError("timeseries.csv must contain ctrl_1 and ctrl_2 columns")

    if "phase" not in df.columns:
        print("Warning: no phase column found. Mocap replay will be disabled.")
        args.use_mocap_replay = False

    if "t" not in df.columns:
        print("Warning: no t column found. Using fixed viewer delay only.")
        df["t"] = np.arange(len(df), dtype=np.float32)

    df = df.copy()
    df["phase_time"] = compute_phase_time_column(df)

    print_trial_info(trial_dir, meta)

    env = SpiRobGraspEnv(args.base_xml, object_spec=obj_spec, use_viewer=True)
    obs = env.reset()

    if args.fixed_camera:
        try_set_fixed_camera(env)

    try:
        for i, row in df.iterrows():
            ctrl = np.array(
                [
                    float(row.get("ctrl_1", 0.0)),
                    float(row.get("ctrl_2", 0.0)),
                ],
                dtype=np.float32,
            )

            if args.use_mocap_replay and hasattr(env, "default_mocap_pos"):
                phase = str(row.get("phase", "pack"))
                phase_time = float(row.get("phase_time", 0.0))
                mocap_pos = base_pull_offset(
                    env.default_mocap_pos,
                    phase,
                    phase_time,
                    args.pull_time,
                )
                env.set_mocap_pos(mocap_pos)

            if i < len(df) - 1:
                t_now = float(df.iloc[i]["t"])
                t_next = float(df.iloc[i + 1]["t"])
                dt_nominal = max(t_next - t_now, 1e-4)
            else:
                dt_nominal = 0.005

            n_substeps = max(1, int(round(dt_nominal / max(env.dt, 1e-6))))
            obs = env.step(ctrl, n_substeps=n_substeps)

            time.sleep(args.viewer_sleep)

        print("Replay finished.")
        if args.auto_exit:
            print("Closing viewer in 1 second...")
            time.sleep(1.0)
        else:
            print("Close the MuJoCo window manually, or press Ctrl+C in the terminal.")

    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"Warning during env.close(): {e}")

        if args.auto_exit:
            os._exit(0)


if __name__ == "__main__":
    main()