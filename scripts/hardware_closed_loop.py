#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np
import torch

from spirob.controller import InterventionPolicy
from spirob.data import load_stats
from spirob.hardware import JsonlImuStream, SocketCanMotorInterface
from spirob.model import ModelConfig, SpiRobTinyModel


def row_from_frame(frame: dict) -> dict:
    row = {
        "ctrl_1": float(frame.get("ctrl_1", 0.0)),
        "ctrl_2": float(frame.get("ctrl_2", 0.0)),
        "actuator_force_1": float(frame.get("actuator_force", [0.0, 0.0])[0]),
        "actuator_force_2": float(frame.get("actuator_force", [0.0, 0.0])[1]),
        "tendon_length_1": float(frame.get("tendon_length", [0.0, 0.0])[0]),
        "tendon_length_2": float(frame.get("tendon_length", [0.0, 0.0])[1]),
        "tendon_velocity_1": float(frame.get("tendon_velocity", [0.0, 0.0])[0]),
        "tendon_velocity_2": float(frame.get("tendon_velocity", [0.0, 0.0])[1]),
        "contact": int(bool(frame.get("contact", False))),
        "object_x": float(frame.get("object_x", 0.0)),
        "object_z": float(frame.get("object_z", 0.0)),
    }
    imu_acc = frame["imu_acc"]
    imu_gyro = frame["imu_gyro"]
    for i in range(len(imu_acc)):
        row[f"imu_gyro_B{i}_x"] = float(imu_gyro[i][0])
        row[f"imu_gyro_B{i}_y"] = float(imu_gyro[i][1])
        row[f"imu_gyro_B{i}_z"] = float(imu_gyro[i][2])
        row[f"imu_acc_B{i}_x"] = float(imu_acc[i][0])
        row[f"imu_acc_B{i}_y"] = float(imu_acc[i][1])
        row[f"imu_acc_B{i}_z"] = float(imu_acc[i][2])
    return row


def tensors_from_ring(ring, stats):
    rows = list(ring)
    T = len(rows)
    n_imus = len([c for c in rows[0].keys() if c.startswith("imu_acc_B") and c.endswith("_x")])
    imu = np.zeros((1, T, n_imus, 6), dtype=np.float32)
    for t, row in enumerate(rows):
        for i in range(n_imus):
            imu[0, t, i, 0] = row[f"imu_gyro_B{i}_x"]
            imu[0, t, i, 1] = row[f"imu_gyro_B{i}_y"]
            imu[0, t, i, 2] = row[f"imu_gyro_B{i}_z"]
            imu[0, t, i, 3] = row[f"imu_acc_B{i}_x"]
            imu[0, t, i, 4] = row[f"imu_acc_B{i}_y"]
            imu[0, t, i, 5] = row[f"imu_acc_B{i}_z"]
    mask = np.ones((1, T, n_imus), dtype=np.float32)
    actuator = np.stack(
        [
            [r["ctrl_1"], r["ctrl_2"], r["actuator_force_1"], r["actuator_force_2"], r["tendon_length_1"], r["tendon_length_2"], r["tendon_velocity_1"], r["tendon_velocity_2"]]
            for r in rows
        ],
        axis=0,
    )[None, ...].astype(np.float32)
    imu = (imu - stats.imu_mean.reshape(1, 1, 1, -1)) / np.maximum(stats.imu_std.reshape(1, 1, 1, -1), 1e-6)
    actuator = (actuator - stats.act_mean.reshape(1, 1, -1)) / np.maximum(stats.act_std.reshape(1, 1, -1), 1e-6)
    return torch.from_numpy(imu), torch.from_numpy(mask), torch.from_numpy(actuator)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--jsonl", required=True, help="Live IMU stream replay file or FIFO")
    ap.add_argument("--can-channel", default="can0")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--window-steps", type=int, default=80)
    ap.add_argument("--decision-every", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cpu")
    stats = load_stats(args.stats)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SpiRobTinyModel(ModelConfig(**ckpt.get("config", {})))
    model.load_state_dict(ckpt["model"])
    model.eval()

    ring = deque(maxlen=args.window_steps)
    policy = InterventionPolicy()
    motor = None if args.dry_run else SocketCanMotorInterface(channel=args.can_channel)
    stream = JsonlImuStream(args.jsonl)

    if motor is not None:
        motor.enable(1)
        motor.enable(2)

    try:
        for step_idx, frame in enumerate(stream):
            row = row_from_frame(frame)
            ring.append(row)
            if len(ring) < args.window_steps or step_idx % args.decision_every != 0:
                continue

            imu, mask, actuator = tensors_from_ring(ring, stats)
            with torch.no_grad():
                out = model(imu, mask, actuator)
            np_out = {k: v.detach().cpu().numpy() for k, v in out.items()}

            class ObsLike:
                contact = bool(frame.get("contact", False))

            action = policy.decide(np_out, ObsLike(), phase="tighten")
            if action == "continue":
                cmd = (0.25, 0.75)
            elif action == "tighten":
                cmd = (0.15, 0.95)
            elif action == "abort":
                cmd = (0.05, 0.05)
            else:
                cmd = (0.80, 0.20)

            if motor is not None:
                motor.send_control(1, cmd[0], 0.0, 30.0, 3.0, 0.0)
                motor.send_control(2, cmd[1], 0.0, 30.0, 3.0, 0.0)
            else:
                print(json.dumps({"t": frame.get("t", 0.0), "action": action, "cmd": cmd}))
    finally:
        if motor is not None:
            motor.disable(1)
            motor.disable(2)
            motor.close()


if __name__ == "__main__":
    main()
