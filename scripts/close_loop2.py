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
import random
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from spirob.object_library import sample_object, workspace_positions
from spirob.sim_env import SpiRobGraspEnv
from spirob.controller import OpenLoopGraspController, base_pull_offset
from spirob.labels import compute_trial_summary
from spirob.model import ModelConfig, SpiRobTinyModel
from spirob.data import load_stats


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


def build_window(
    history: list[dict],
    window_steps: int,
    imu_indices: list[int],
    actuator_columns: list[str],
    stats=None,
):
    n_imu = len(imu_indices)
    act_dim = len(actuator_columns)

    imu = np.zeros((window_steps, n_imu, 6), dtype=np.float32)
    mask = np.zeros((window_steps, n_imu), dtype=np.float32)
    actuator = np.zeros((window_steps, act_dim), dtype=np.float32)

    use_rows = history[-window_steps:]
    offset = window_steps - len(use_rows)

    for t, row in enumerate(use_rows, start=offset):
        for j, idx in enumerate(imu_indices):
            imu[t, j, 0] = row.get(f"imu_gyro_B{idx}_x", 0.0)
            imu[t, j, 1] = row.get(f"imu_gyro_B{idx}_y", 0.0)
            imu[t, j, 2] = row.get(f"imu_gyro_B{idx}_z", 0.0)
            imu[t, j, 3] = row.get(f"imu_acc_B{idx}_x", 0.0)
            imu[t, j, 4] = row.get(f"imu_acc_B{idx}_y", 0.0)
            imu[t, j, 5] = row.get(f"imu_acc_B{idx}_z", 0.0)
            mask[t, j] = 1.0

        for k, col in enumerate(actuator_columns):
            actuator[t, k] = row.get(col, 0.0)

    if stats is not None:
        imu = (imu - stats.imu_mean.reshape(1, 1, -1)) / np.maximum(stats.imu_std.reshape(1, 1, -1), 1e-6)
        actuator = (actuator - stats.act_mean.reshape(1, -1)) / np.maximum(stats.act_std.reshape(1, -1), 1e-6)

    return imu, mask, actuator


def obs_to_row(obs, controller) -> dict:
    row = {
        "t": float(obs.t),
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

    return row


def choose_intervention(model_out: dict) -> str:
    action_logits = model_out["action_logits"]
    if action_logits.ndim == 2:
        action_id = int(torch.argmax(action_logits, dim=-1).item())
    else:
        action_id = int(torch.argmax(action_logits).item())

    id_to_action = {
        0: "none",
        1: "reclose",
        2: "slow_pull",
        3: "abort",
    }
    return id_to_action.get(action_id, "none")


def apply_intervention_to_controller(controller, intervention: str, ignore_abort: bool = True) -> None:
    if intervention == "abort":
        if ignore_abort:
            return
        controller.phase = "done"
        return

    if intervention == "slow_pull":
        if hasattr(controller, "pull_time"):
            controller.pull_time = float(controller.pull_time) * 1.25
        return

    if intervention == "reclose":
        if controller.phase in {"pack", "unpack", "reach", "wrap", "tighten", "pull"}:
            controller.phase = "wrap"
            if hasattr(controller, "phase_time"):
                controller.phase_time = 0.0


def maybe_sleep(seconds: float) -> None:
    if seconds <= 0:
        return
    time.sleep(seconds)


def close_env_safely(env) -> None:
    if env is None:
        return
    try:
        env.close()
    except Exception:
        pass


def print_trial_result(trial_id: int, obj, summary) -> None:
    status = "SUCCESS" if int(summary.success) == 1 else "FAILURE"
    print(
        f"\nTrial {trial_id:04d} | {status} | "
        f"family={obj.family} | size={obj.size:.5f} | "
        f"quality={summary.quality:.4f} | "
        f"attach_ratio={summary.attached_ratio_pull:.4f} | "
        f"hold_time={summary.hold_time:.4f} | "
        f"slip_penalty={summary.slip_penalty:.4f} | "
        f"lift_distance={summary.lift_distance:.4f}",
        flush=True,
    )


def make_size_bins(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="object")
    try:
        bins = pd.qcut(df["size"], q=3, labels=["small", "medium", "large"], duplicates="drop")
        return bins.astype(str)
    except Exception:
        return pd.Series(["all"] * len(df), index=df.index, dtype="object")


def save_success_failure_analysis(summary_df: pd.DataFrame, outdir: Path) -> None:
    analysis_dir = outdir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df = summary_df.copy()
    df["outcome"] = df["success"].map({1: "success", 0: "failure"})
    df["size_bin"] = make_size_bins(df)

    # Tables
    by_shape = (
        df.groupby(["family", "outcome"], as_index=False)
        .agg(
            num_trials=("trial_id", "count"),
            mean_quality=("quality", "mean"),
            mean_hold_time=("hold_time", "mean"),
            mean_slip_penalty=("slip_penalty", "mean"),
            mean_attach_ratio=("attached_ratio_pull", "mean"),
        )
        .sort_values(["family", "outcome"])
    )
    by_shape.to_csv(analysis_dir / "success_failure_by_shape.csv", index=False)

    by_size = (
        df.groupby(["size_bin", "outcome"], as_index=False)
        .agg(
            num_trials=("trial_id", "count"),
            mean_size=("size", "mean"),
            mean_quality=("quality", "mean"),
            mean_hold_time=("hold_time", "mean"),
            mean_slip_penalty=("slip_penalty", "mean"),
            mean_attach_ratio=("attached_ratio_pull", "mean"),
        )
        .sort_values(["size_bin", "outcome"])
    )
    by_size.to_csv(analysis_dir / "success_failure_by_size.csv", index=False)

    by_location = (
        df.groupby(["pose_x", "pose_z", "outcome"], as_index=False)
        .agg(
            num_trials=("trial_id", "count"),
            mean_quality=("quality", "mean"),
            mean_hold_time=("hold_time", "mean"),
            mean_slip_penalty=("slip_penalty", "mean"),
            mean_attach_ratio=("attached_ratio_pull", "mean"),
        )
        .sort_values(["pose_z", "pose_x", "outcome"])
    )
    by_location.to_csv(analysis_dir / "success_failure_by_location.csv", index=False)

    failed_only = (
        df[df["success"] == 0][
            [
                "trial_id",
                "family",
                "size",
                "pose_x",
                "pose_z",
                "quality",
                "pull_force_proxy",
                "hold_time",
                "slip_penalty",
                "attached_ratio_pull",
                "num_decisions",
            ]
        ]
        .sort_values(["family", "size", "pose_z", "pose_x"])
    )
    failed_only.to_csv(analysis_dir / "failed_trial_characteristics.csv", index=False)

    # Shape plot
    shape_plot = (
        df.groupby(["family", "outcome"])["trial_id"]
        .count()
        .unstack(fill_value=0)
    )
    ax = shape_plot.plot(kind="bar", figsize=(8, 4.5))
    ax.set_title("Success / Failure by Shape")
    ax.set_xlabel("Shape / Family")
    ax.set_ylabel("Number of Trials")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(analysis_dir / "success_failure_by_shape.png", dpi=200)
    plt.close()

    # Size plot
    size_plot = (
        df.groupby(["size_bin", "outcome"])["trial_id"]
        .count()
        .unstack(fill_value=0)
        .reindex(["small", "medium", "large"], fill_value=0)
    )
    ax = size_plot.plot(kind="bar", figsize=(7, 4.5))
    ax.set_title("Success / Failure by Size")
    ax.set_xlabel("Size Bin")
    ax.set_ylabel("Number of Trials")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(analysis_dir / "success_failure_by_size.png", dpi=200)
    plt.close()

    # Workspace map
    plt.figure(figsize=(7, 6))
    succ = df[df["success"] == 1]
    fail = df[df["success"] == 0]

    if not succ.empty:
        plt.scatter(succ["pose_x"], succ["pose_z"], marker="o", label="Success", alpha=0.8)
    if not fail.empty:
        plt.scatter(fail["pose_x"], fail["pose_z"], marker="x", label="Failure", alpha=0.8)

    plt.xlabel("Workspace X")
    plt.ylabel("Workspace Z")
    plt.title("Workspace Map of Success / Failure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / "workspace_success_failure_map.png", dpi=200)
    plt.close()

    # Location table summarized by point
    location_summary = (
        df.groupby(["pose_x", "pose_z"], as_index=False)
        .agg(
            num_trials=("trial_id", "count"),
            num_success=("success", "sum"),
            num_failure=("success", lambda s: int((s == 0).sum())),
            mean_quality=("quality", "mean"),
        )
        .sort_values(["pose_z", "pose_x"])
    )
    location_summary.to_csv(analysis_dir / "location_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-xml", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--outdir", required=True)

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

    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--window-steps", type=int, default=80)
    ap.add_argument("--decision-every", type=int, default=20)
    ap.add_argument("--decision-start-step", type=int, default=150)
    ap.add_argument("--trial-seconds", type=float, default=15.0)
    ap.add_argument("--hz", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--viewer-sleep", type=float, default=0.06)
    ap.add_argument("--allow-abort", action="store_true")
    ap.add_argument("--no-fixed-camera", action="store_true")
    ap.add_argument("--hold-on-finish", type=float, default=0.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    actuator_columns = resolve_actuator_columns(args.actuator_preset, args.actuator_columns)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    trials_dir = outdir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(args.stats)
    ckpt = torch.load(args.checkpoint, map_location=device)

    config_dict = ckpt.get("config", {})
    try:
        model = SpiRobTinyModel(ModelConfig(**config_dict)).to(device)
    except TypeError:
        model = SpiRobTinyModel(ModelConfig()).to(device)

    if model.config.act_dim != len(actuator_columns):
        raise ValueError(
            f"Checkpoint expects act_dim={model.config.act_dim}, "
            f"but selected actuator columns give {len(actuator_columns)}. Retrain with matching act_dim."
        )

    model.load_state_dict(ckpt["model"])
    model.eval()

    positions = workspace_positions()
    summary_rows = []

    try:
        for trial_id in tqdm(range(args.num_trials), desc="closed-loop"):
            env = None
            try:
                obj = sample_object(rng, positions=positions)
                env = SpiRobGraspEnv(args.base_xml, object_spec=obj, use_viewer=args.viewer)
                obs = env.reset()

                available = list(range(int(obs.imu_acc.shape[0])))
                imu_indices = resolve_imu_indices(available, args.imu_preset, args.imu_indices)

                if (
                    args.viewer
                    and not args.no_fixed_camera
                    and hasattr(env, "viewer")
                    and env.viewer is not None
                ):
                    try:
                        import mujoco
                        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "camera1")
                        if cam_id >= 0:
                            env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                            env.viewer.cam.fixedcamid = int(cam_id)
                            if hasattr(env.viewer.cam, "trackbodyid"):
                                env.viewer.cam.trackbodyid = -1
                            print(f"Using fixed XML camera: camera1 (id={cam_id})")
                    except Exception as e:
                        print(f"Could not set fixed XML camera: {e}")

                controller = OpenLoopGraspController()
                controller.reset(obj.pose_x, obj.pose_z)

                trial_path = trials_dir / f"trial_{trial_id:04d}"
                trial_path.mkdir(parents=True, exist_ok=True)

                dt_nominal = 1.0 / args.hz
                steps = int(args.trial_seconds * args.hz)

                rows: list[dict] = []
                history: list[dict] = []
                decisions: list[dict] = []

                for step_idx in range(steps):
                    intervention = "none"

                    should_decide = (
                        step_idx >= max(args.window_steps, args.decision_start_step)
                        and step_idx % args.decision_every == 0
                    )

                    if should_decide:
                        imu, mask, actuator = build_window(
                            history=history,
                            window_steps=args.window_steps,
                            imu_indices=imu_indices,
                            actuator_columns=actuator_columns,
                            stats=stats,
                        )

                        batch_imu = torch.from_numpy(imu).unsqueeze(0).to(device)
                        batch_mask = torch.from_numpy(mask).unsqueeze(0).to(device)
                        batch_actuator = torch.from_numpy(actuator).unsqueeze(0).to(device)

                        with torch.no_grad():
                            model_out = model(batch_imu, batch_mask, batch_actuator)

                        intervention = choose_intervention(model_out)
                        apply_intervention_to_controller(
                            controller,
                            intervention,
                            ignore_abort=not args.allow_abort,
                        )

                        decisions.append(
                            {
                                "t": float(obs.t),
                                "step": int(step_idx),
                                "intervention": intervention,
                                "success_prob": float(model_out["success_prob"].item()),
                                "quality_pred": float(model_out["quality"].item()),
                            }
                        )

                    ctrl = controller.update(obs, dt=dt_nominal, intervention=intervention)
                    mocap_pos = base_pull_offset(
                        env.default_mocap_pos,
                        controller.phase,
                        controller.phase_time,
                        controller.pull_time,
                    )
                    env.set_mocap_pos(mocap_pos)

                    n_substeps = max(1, int(round(dt_nominal / max(env.dt, 1e-6))))
                    obs = env.step(ctrl, n_substeps=n_substeps)

                    if args.viewer:
                        maybe_sleep(args.viewer_sleep)

                    row = obs_to_row(obs, controller)
                    row["intervention"] = intervention
                    rows.append(row)
                    history.append(row)

                    if controller.phase == "done":
                        if args.viewer and args.hold_on_finish > 0:
                            maybe_sleep(args.hold_on_finish)
                        break

                df = pd.DataFrame(rows)
                df.to_csv(trial_path / "timeseries.csv", index=False)

                with open(trial_path / "decisions.json", "w", encoding="utf-8") as f:
                    json.dump(decisions, f, indent=2)

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
                    "num_decisions": len(decisions),
                    "imu_indices": imu_indices,
                    "actuator_columns": actuator_columns,
                }

                with open(trial_path / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                summary_rows.append(
                    {
                        "trial_id": trial_id,
                        "family": obj.family,
                        "size": obj.size,
                        "mass": obj.mass,
                        "pose_x": obj.pose_x,
                        "pose_z": obj.pose_z,
                        "yaw": obj.yaw,
                        "success": summary.success,
                        "quality": summary.quality,
                        "pull_force_proxy": summary.pull_force_proxy,
                        "hold_time": summary.hold_time,
                        "slip_penalty": summary.slip_penalty,
                        "attached_ratio_pull": summary.attached_ratio_pull,
                        "lift_distance": summary.lift_distance,
                        "num_decisions": len(decisions),
                    }
                )

                print_trial_result(trial_id, obj, summary)

            finally:
                close_env_safely(env)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Closed viewer and exiting cleanly.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "summary.csv", index=False)

    if not summary_df.empty:
        save_success_failure_analysis(summary_df, outdir)

    print("\nRun finished. Results saved.", flush=True)

    # Clean termination when viewer was used
    if args.viewer:
        os._exit(0)


if __name__ == "__main__":
    main()