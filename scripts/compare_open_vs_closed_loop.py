#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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


def close_env_safely(env) -> None:
    if env is None:
        return
    try:
        env.close()
    except Exception:
        pass


def make_size_bins(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="object")
    try:
        bins = pd.qcut(df["size"], q=3, labels=["small", "medium", "large"], duplicates="drop")
        return bins.astype(str)
    except Exception:
        return pd.Series(["all"] * len(df), index=df.index, dtype="object")


def load_model(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device)
    config_dict = ckpt.get("config", {})
    try:
        model = SpiRobTinyModel(ModelConfig(**config_dict)).to(device)
    except TypeError:
        model = SpiRobTinyModel(ModelConfig()).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def run_policy(
    policy_name: str,
    num_trials: int,
    objects: list,
    args,
    outdir: Path,
    stats,
    device: torch.device,
    model,
    actuator_columns: list[str],
) -> pd.DataFrame:
    policy_dir = outdir / policy_name
    trials_dir = policy_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for trial_id in tqdm(range(num_trials), desc=policy_name):
        env = None
        try:
            obj = objects[trial_id]
            env = SpiRobGraspEnv(args.base_xml, object_spec=obj, use_viewer=False)
            obs = env.reset()

            available = list(range(int(obs.imu_acc.shape[0])))
            imu_indices = resolve_imu_indices(available, args.imu_preset, args.imu_indices)

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

                if policy_name == "closed_loop":
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

                row = obs_to_row(obs, controller)
                row["intervention"] = intervention
                rows.append(row)
                history.append(row)

                if controller.phase == "done":
                    break

            df = pd.DataFrame(rows)
            df.to_csv(trial_path / "timeseries.csv", index=False)

            with open(trial_path / "decisions.json", "w", encoding="utf-8") as f:
                json.dump(decisions, f, indent=2)

            summary = compute_trial_summary(df)
            meta = {
                "trial_id": trial_id,
                "policy": policy_name,
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
                    "policy": policy_name,
                    "family": obj.family,
                    "size": obj.size,
                    "mass": obj.mass,
                    "pose_x": obj.pose_x,
                    "pose_z": obj.pose_z,
                    "yaw": obj.yaw,
                    "success": int(summary.success),
                    "quality": float(summary.quality),
                    "pull_force_proxy": float(summary.pull_force_proxy),
                    "hold_time": float(summary.hold_time),
                    "slip_penalty": float(summary.slip_penalty),
                    "attached_ratio_pull": float(summary.attached_ratio_pull),
                    "lift_distance": float(summary.lift_distance),
                    "num_decisions": len(decisions),
                }
            )

        finally:
            close_env_safely(env)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(policy_dir / "summary.csv", index=False)
    return summary_df


def save_comparison_tables(df_all: pd.DataFrame, outdir: Path) -> None:
    plots_dir = outdir / "comparison_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    overall = (
        df_all.groupby("policy", as_index=False)
        .agg(
            num_trials=("trial_id", "count"),
            num_success=("success", "sum"),
            success_rate=("success", "mean"),
            mean_quality=("quality", "mean"),
            mean_hold_time=("hold_time", "mean"),
            mean_slip_penalty=("slip_penalty", "mean"),
            mean_attach_ratio=("attached_ratio_pull", "mean"),
            mean_lift_distance=("lift_distance", "mean"),
            mean_num_decisions=("num_decisions", "mean"),
        )
    )
    overall["success_rate_percent"] = 100.0 * overall["success_rate"]
    overall.to_csv(plots_dir / "overall_comparison.csv", index=False)

    by_family = (
        df_all.groupby(["policy", "family"], as_index=False)
        .agg(
            num_trials=("trial_id", "count"),
            num_success=("success", "sum"),
            success_rate=("success", "mean"),
            mean_quality=("quality", "mean"),
        )
    )
    by_family["success_rate_percent"] = 100.0 * by_family["success_rate"]
    by_family.to_csv(plots_dir / "comparison_by_family.csv", index=False)

    df_tmp = df_all.copy()
    df_tmp["size_bin"] = make_size_bins(df_tmp)
    by_size = (
        df_tmp.groupby(["policy", "size_bin"], as_index=False)
        .agg(
            num_trials=("trial_id", "count"),
            num_success=("success", "sum"),
            success_rate=("success", "mean"),
            mean_size=("size", "mean"),
            mean_quality=("quality", "mean"),
        )
    )
    by_size["success_rate_percent"] = 100.0 * by_size["success_rate"]
    by_size.to_csv(plots_dir / "comparison_by_size.csv", index=False)

    by_location = (
        df_all.groupby(["policy", "pose_x", "pose_z"], as_index=False)
        .agg(
            num_trials=("trial_id", "count"),
            num_success=("success", "sum"),
            success_rate=("success", "mean"),
            mean_quality=("quality", "mean"),
        )
    )
    by_location["success_rate_percent"] = 100.0 * by_location["success_rate"]
    by_location.to_csv(plots_dir / "comparison_by_location.csv", index=False)

    failed = (
        df_all[df_all["success"] == 0]
        .sort_values(["policy", "family", "size"])
        .reset_index(drop=True)
    )
    failed.to_csv(plots_dir / "failed_trials_comparison.csv", index=False)


def plot_comparison(df_all: pd.DataFrame, outdir: Path) -> None:
    plots_dir = outdir / "comparison_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall success rate
    overall = (
        df_all.groupby("policy", as_index=False)
        .agg(success_rate=("success", "mean"))
    )
    overall["success_rate_percent"] = 100.0 * overall["success_rate"]

    plt.figure(figsize=(6.5, 4.8))
    bars = plt.bar(overall["policy"], overall["success_rate_percent"])
    plt.ylabel("Success Rate (%)")
    plt.title("Open-Loop Baseline vs Model-Guided Closed-Loop")
    for bar, v in zip(bars, overall["success_rate_percent"]):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 1.0, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(plots_dir / "overall_success_rate_comparison.png", dpi=220)
    plt.close()

    # 2. Success by family
    fam = (
        df_all.groupby(["policy", "family"], as_index=False)
        .agg(success_rate=("success", "mean"))
    )
    fam["success_rate_percent"] = 100.0 * fam["success_rate"]
    fam_pivot = fam.pivot(index="family", columns="policy", values="success_rate_percent").fillna(0.0)
    fam_pivot = fam_pivot.sort_index()

    ax = fam_pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Family")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(plots_dir / "success_by_family_comparison.png", dpi=220)
    plt.close()

    # 3. Success by size
    df_tmp = df_all.copy()
    df_tmp["size_bin"] = make_size_bins(df_tmp)
    siz = (
        df_tmp.groupby(["policy", "size_bin"], as_index=False)
        .agg(success_rate=("success", "mean"))
    )
    siz["success_rate_percent"] = 100.0 * siz["success_rate"]
    siz_pivot = siz.pivot(index="size_bin", columns="policy", values="success_rate_percent").fillna(0.0)

    desired_order = [x for x in ["small", "medium", "large", "all"] if x in siz_pivot.index]
    siz_pivot = siz_pivot.reindex(desired_order)

    ax = siz_pivot.plot(kind="bar", figsize=(7.5, 5))
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Size Bin")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plots_dir / "success_by_size_comparison.png", dpi=220)
    plt.close()

    # 4. Workspace maps side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax, policy in zip(axes, ["open_loop", "closed_loop"]):
        sub = df_all[df_all["policy"] == policy]
        succ = sub[sub["success"] == 1]
        fail = sub[sub["success"] == 0]

        if not succ.empty:
            ax.scatter(succ["pose_x"], succ["pose_z"], marker="o", alpha=0.8, label="Success")
        if not fail.empty:
            ax.scatter(fail["pose_x"], fail["pose_z"], marker="x", alpha=0.8, label="Failure")

        ax.set_title(policy.replace("_", " ").title())
        ax.set_xlabel("Workspace X")
        ax.set_ylabel("Workspace Z")
        ax.legend()

    plt.suptitle("Workspace Success / Failure Comparison")
    plt.tight_layout()
    plt.savefig(plots_dir / "workspace_map_comparison.png", dpi=220)
    plt.close()

    # 5. Failure characteristics comparison
    fail_stats = (
        df_all[df_all["success"] == 0]
        .groupby("policy", as_index=False)
        .agg(
            mean_quality=("quality", "mean"),
            mean_hold_time=("hold_time", "mean"),
            mean_slip_penalty=("slip_penalty", "mean"),
            mean_attach_ratio=("attached_ratio_pull", "mean"),
        )
    )

    if not fail_stats.empty:
        metrics = [
            "mean_quality",
            "mean_hold_time",
            "mean_slip_penalty",
            "mean_attach_ratio",
        ]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes = axes.ravel()

        for ax, metric in zip(axes, metrics):
            ax.bar(fail_stats["policy"], fail_stats[metric])
            ax.set_title(metric.replace("_", " ").title())
            ax.tick_params(axis="x", rotation=15)

        plt.suptitle("Failed-Trial Characteristics")
        plt.tight_layout()
        plt.savefig(plots_dir / "failed_trial_characteristics_comparison.png", dpi=220)
        plt.close()


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

    ap.add_argument("--num-trials", type=int, default=200)
    ap.add_argument("--window-steps", type=int, default=80)
    ap.add_argument("--decision-every", type=int, default=20)
    ap.add_argument("--decision-start-step", type=int, default=150)
    ap.add_argument("--trial-seconds", type=float, default=6.0)
    ap.add_argument("--hz", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--allow-abort", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    stats = load_stats(args.stats)
    actuator_columns = resolve_actuator_columns(args.actuator_preset, args.actuator_columns)
    model = load_model(args.checkpoint, device)

    if model.config.act_dim != len(actuator_columns):
        raise ValueError(
            f"Checkpoint expects act_dim={model.config.act_dim}, "
            f"but selected actuator columns give {len(actuator_columns)}. Retrain with matching act_dim."
        )

    # Pre-sample exactly the same objects for both policies
    positions = workspace_positions()
    objects = [sample_object(rng, positions=positions) for _ in range(args.num_trials)]

    print("Running open-loop baseline...")
    df_open = run_policy(
        policy_name="open_loop",
        num_trials=args.num_trials,
        objects=objects,
        args=args,
        outdir=outdir,
        stats=stats,
        device=device,
        model=model,
        actuator_columns=actuator_columns,
    )

    print("Running model-guided closed-loop...")
    df_closed = run_policy(
        policy_name="closed_loop",
        num_trials=args.num_trials,
        objects=objects,
        args=args,
        outdir=outdir,
        stats=stats,
        device=device,
        model=model,
        actuator_columns=actuator_columns,
    )

    df_all = pd.concat([df_open, df_closed], ignore_index=True)
    df_all.to_csv(outdir / "combined_summary.csv", index=False)

    save_comparison_tables(df_all, outdir)
    plot_comparison(df_all, outdir)

    print(f"Saved combined results to: {outdir}")
    print(f"Open-loop success rate:   {df_open['success'].mean() * 100.0:.2f}%")
    print(f"Closed-loop success rate: {df_closed['success'].mean() * 100.0:.2f}%")


if __name__ == "__main__":
    main()