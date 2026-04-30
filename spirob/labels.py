from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


PHASES = ("pack", "unpack", "reach", "wrap", "tighten", "pull", "done")
ACTIONS = ("continue", "tighten", "abort", "retry")

ACTION_TO_ID = {name: i for i, name in enumerate(ACTIONS)}
ID_TO_ACTION = {i: name for i, name in enumerate(ACTIONS)}

PHASE_TO_ID = {name: i for i, name in enumerate(PHASES)}
ID_TO_PHASE = {i: name for i, name in enumerate(PHASES)}


@dataclass
class TrialSummary:
    success: int
    quality: float
    pull_force_proxy: float
    hold_time: float
    slip_penalty: float
    attached_ratio_pull: float
    lift_distance: float


def _ensure_phase(df: pd.DataFrame) -> pd.DataFrame:
    if "phase" in df.columns:
        out = df.copy()
        out["phase"] = out["phase"].astype(str)
        return out

    out = df.copy()
    t = out["t"].to_numpy(float) if "t" in out.columns else np.arange(len(out), dtype=float)

    if len(t) == 0:
        out["phase"] = pd.Series(dtype=object)
        return out

    total = max(float(t[-1] - t[0]), 1e-6)
    alpha = (t - t[0]) / total

    # Fallback phase schedule for logs that do not already contain a phase column.
    phase = np.full(len(out), "pull", dtype=object)
    phase[alpha < 0.10] = "pack"
    phase[(alpha >= 0.10) & (alpha < 0.22)] = "unpack"
    phase[(alpha >= 0.22) & (alpha < 0.48)] = "reach"
    phase[(alpha >= 0.48) & (alpha < 0.65)] = "wrap"
    phase[(alpha >= 0.65) & (alpha < 0.82)] = "tighten"
    phase[alpha >= 0.82] = "pull"

    out["phase"] = phase
    return out


def add_trial_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_phase(df.copy())
    if len(out) == 0:
        return out

    if "t" not in out.columns:
        out["t"] = np.arange(len(out), dtype=float)

    out["dt"] = out["t"].diff().fillna(0.0).clip(lower=0.0)

    if {"tip_x", "tip_y", "tip_z", "obj_x", "obj_y", "obj_z"}.issubset(out.columns):
        out["tip_obj_dist"] = np.sqrt(
            (out["tip_x"] - out["obj_x"]) ** 2
            + (out["tip_y"] - out["obj_y"]) ** 2
            + (out["tip_z"] - out["obj_z"]) ** 2
        )
    elif {"tip_x", "tip_z", "obj_x", "obj_z"}.issubset(out.columns):
        out["tip_obj_dist"] = np.sqrt(
            (out["tip_x"] - out["obj_x"]) ** 2
            + (out["tip_z"] - out["obj_z"]) ** 2
        )
    else:
        out["tip_obj_dist"] = 0.0

    out["contact_bool"] = out["contact"].astype(float) if "contact" in out.columns else 0.0

    if "contact_count" not in out.columns:
        out["contact_count"] = out["contact_bool"]

    if "contact_force_proxy" not in out.columns:
        af1 = out["actuator_force_1"] if "actuator_force_1" in out.columns else 0.0
        af2 = out["actuator_force_2"] if "actuator_force_2" in out.columns else 0.0
        out["contact_force_proxy"] = out["contact_bool"] * (np.abs(af1) + np.abs(af2))

    if "obj_z" in out.columns:
        out["obj_z_rel"] = out["obj_z"] - float(out["obj_z"].iloc[0])
    else:
        out["obj_z_rel"] = 0.0

    if "tendon_length_1" in out.columns and "tendon_length_2" in out.columns:
        out["tendon_span"] = np.abs(out["tendon_length_1"] - out["tendon_length_2"])
    else:
        out["tendon_span"] = 0.0

    return out


def window_phase_label(wdf: pd.DataFrame) -> str:
    wdf = _ensure_phase(wdf)
    if len(wdf) == 0:
        return "pack"

    phase = str(wdf["phase"].mode(dropna=True).iloc[0])
    if phase not in PHASE_TO_ID:
        return "pack"
    return phase


def compute_trial_summary(df: pd.DataFrame) -> TrialSummary:
    df = add_trial_features(df)
    if len(df) == 0:
        return TrialSummary(0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    pull_df = df[df["phase"] == "pull"]
    if len(pull_df) == 0:
        pull_df = df.iloc[max(0, int(0.75 * len(df))):]

    total_pull_time = (
        float(pull_df["dt"].sum()) if "dt" in pull_df.columns else float(max(len(pull_df) - 1, 0))
    )
    contact_ratio = float(df["contact_bool"].mean()) if "contact_bool" in df.columns else 0.0

    close_thresh = float(np.clip(df["tip_obj_dist"].quantile(0.25) + 0.02, 0.025, 0.08))
    attached_mask = (pull_df["tip_obj_dist"] <= close_thresh).astype(float)
    attached_ratio_pull = float(attached_mask.mean()) if len(pull_df) else 0.0
    hold_time = (
        float((attached_mask * pull_df["dt"]).sum())
        if len(pull_df) and "dt" in pull_df.columns
        else 0.0
    )

    lift_distance = float(df["obj_z_rel"].max()) if "obj_z_rel" in df.columns else 0.0

    late_dist = (
        pull_df["tip_obj_dist"].to_numpy(float)
        if len(pull_df)
        else np.array([0.0], dtype=float)
    )
    slip_penalty = float(
        np.clip((late_dist[-1] - late_dist.min()) / max(close_thresh, 1e-6), 0.0, 1.0)
    )

    pull_force_proxy = (
        float(
            np.mean(
                np.abs(pull_df.get("actuator_force_1", 0.0))
                + np.abs(pull_df.get("actuator_force_2", 0.0))
            )
        )
        if len(pull_df)
        else 0.0
    )
    tightness_bonus = float(np.tanh(max(pull_force_proxy, 0.0) / 4.0))

    quality = (
        0.45 * attached_ratio_pull
        + 0.20 * contact_ratio
        + 0.15 * np.clip(lift_distance / 0.05, 0.0, 1.0)
        + 0.10 * tightness_bonus
        + 0.10 * (1.0 - slip_penalty)
    )
    quality = float(np.clip(quality, 0.0, 1.0))
    success = int(attached_ratio_pull >= 0.45 and contact_ratio >= 0.10 and quality >= 0.45)

    return TrialSummary(
        success=success,
        quality=quality,
        pull_force_proxy=float(pull_force_proxy),
        hold_time=hold_time if total_pull_time > 0 else 0.0,
        slip_penalty=slip_penalty,
        attached_ratio_pull=attached_ratio_pull,
        lift_distance=lift_distance,
    )


def oracle_intervention_label(wdf: pd.DataFrame, trial_summary: TrialSummary) -> str:
    wdf = add_trial_features(wdf)
    phase = window_phase_label(wdf)

    contact_ratio = float(wdf["contact_bool"].mean()) if "contact_bool" in wdf.columns else 0.0
    dist = float(wdf["tip_obj_dist"].mean()) if "tip_obj_dist" in wdf.columns else 0.0
    force_proxy = float(
        np.mean(
            np.abs(wdf.get("actuator_force_1", 0.0))
            + np.abs(wdf.get("actuator_force_2", 0.0))
        )
    )

    if phase == "pull" and trial_summary.attached_ratio_pull < 0.20:
        return "abort"

    if phase in {"unpack", "reach", "wrap"} and contact_ratio < 0.05 and dist > 0.07:
        return "retry"

    if phase in {"wrap", "tighten", "pull"} and (
        trial_summary.quality < 0.55 or force_proxy < 0.10
    ):
        return "tighten"

    return "continue"