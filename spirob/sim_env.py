from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

import numpy as np

from .object_library import ObjectSpec
from .xml_utils import parse_imu_layout, write_patched_xml


@dataclass
class Observation:
    t: float
    ctrl: np.ndarray
    actuator_force: np.ndarray
    tendon_length: np.ndarray
    tendon_velocity: np.ndarray
    contact: bool
    contact_count: int
    contact_force_proxy: float
    object_pos: np.ndarray
    object_vel: np.ndarray
    tip_pos: np.ndarray
    imu_acc: np.ndarray
    imu_gyro: np.ndarray
    imu_quat: np.ndarray
    body_positions: np.ndarray


class SpiRobGraspEnv:
    def __init__(self, base_xml: str | Path, object_spec: ObjectSpec, use_viewer: bool = False) -> None:
        try:
            import mujoco
        except Exception as exc:
            raise RuntimeError('mujoco is required for SpiRobGraspEnv. Install it with: python -m pip install mujoco') from exc

        self.mujoco = mujoco
        self.base_xml = Path(base_xml)
        self.object_spec = object_spec
        self.use_viewer = use_viewer
        self.tmp_xml = write_patched_xml(self.base_xml, self.object_spec)
        self.model = mujoco.MjModel.from_xml_path(str(self.tmp_xml))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.viewer = None
        self._setup_ids()
        self.default_mocap_pos = np.array(self.data.mocap_pos[self.mocap_id], dtype=np.float32) if self.mocap_id >= 0 else np.zeros(3, dtype=np.float32)
        if self.use_viewer:
            self._init_viewer()
        mujoco.mj_forward(self.model, self.data)

    def _init_viewer(self) -> None:
        try:
            from mujoco import viewer as mjviewer
            self.viewer = mjviewer.launch_passive(self.model, self.data)
        except Exception:
            self.viewer = None

    def _safe_name2id(self, objtype, name: str) -> int:
        return int(self.mujoco.mj_name2id(self.model, objtype, name))

    def _setup_ids(self) -> None:
        mj = self.mujoco
        self.obj_body_id = self._safe_name2id(mj.mjtObj.mjOBJ_BODY, 'O1')
        self.mocap_body_id = self._safe_name2id(mj.mjtObj.mjOBJ_BODY, 'mocapt')
        self.mocap_id = int(self.model.body_mocapid[self.mocap_body_id]) if self.mocap_body_id >= 0 else -1

        self.body_ids = []
        for i in range(24):
            bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, f'B{i}')
            if bid >= 0:
                self.body_ids.append(int(bid))
        self.tip_body_id = self.body_ids[-1] if self.body_ids else self.obj_body_id

        self.acc_sensors = []
        self.gyro_sensors = []
        self.quat_sensors = []
        for i in range(self.model.nsensor):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_SENSOR, i) or ''
            adr = int(self.model.sensor_adr[i])
            dim = int(self.model.sensor_dim[i])
            if name.startswith('imu_acc_B'):
                idx = int(name.replace('imu_acc_B', ''))
                self.acc_sensors.append((idx, adr, dim))
            elif name.startswith('imu_gyro_B'):
                idx = int(name.replace('imu_gyro_B', ''))
                self.gyro_sensors.append((idx, adr, dim))
            elif name.startswith('gt_att_B'):
                idx = int(name.replace('gt_att_B', ''))
                self.quat_sensors.append((idx, adr, dim))
        self.acc_sensors.sort()
        self.gyro_sensors.sort()
        self.quat_sensors.sort()

        self.object_geom_ids = [i for i in range(self.model.ngeom) if int(self.model.geom_bodyid[i]) == self.obj_body_id]
        self.backbone_geom_ids = [
            i for i in range(self.model.ngeom)
            if int(self.model.geom_bodyid[i]) in set(self.body_ids)
        ]

        self.act_ids = []
        for name in ('A_1', 'A_2'):
            aid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                self.act_ids.append(int(aid))
        self.ten_ids = []
        for name in ('t1', 't2'):
            tid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_TENDON, name)
            if tid >= 0:
                self.ten_ids.append(int(tid))

    def reset(self) -> Observation:
        self.mujoco.mj_resetData(self.model, self.data)
        if self.mocap_id >= 0:
            self.data.mocap_pos[self.mocap_id] = self.default_mocap_pos.copy()
        self.mujoco.mj_forward(self.model, self.data)
        return self._make_obs()

    def set_mocap_pos(self, pos) -> None:
        if self.mocap_id >= 0:
            self.data.mocap_pos[self.mocap_id] = np.asarray(pos, dtype=np.float32)
            self.mujoco.mj_forward(self.model, self.data)

    def _set_ctrl(self, ctrl) -> None:
        ctrl = np.asarray(ctrl, dtype=np.float32)
        if self.model.nu <= 0:
            return
        for i in range(min(len(ctrl), self.model.nu)):
            u = float(ctrl[i])
            if hasattr(self.model, 'actuator_ctrllimited') and int(self.model.actuator_ctrllimited[i]) != 0:
                lo, hi = self.model.actuator_ctrlrange[i]
                u = float(np.clip(u, lo, hi))
            self.data.ctrl[i] = u

    def step(self, ctrl, n_substeps: int = 1) -> Observation:
        self._set_ctrl(ctrl)
        for _ in range(max(1, int(n_substeps))):
            self.mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception:
                pass
        return self._make_obs()

    def _collect_sensor_block(self, sensors, out_dim: int) -> np.ndarray:
        if not sensors:
            return np.zeros((0, out_dim), dtype=np.float32)
        out = np.zeros((len(sensors), out_dim), dtype=np.float32)
        for row, (_, adr, dim) in enumerate(sensors):
            vec = np.asarray(self.data.sensordata[adr:adr + dim], dtype=np.float32)
            out[row, :min(len(vec), out_dim)] = vec[:out_dim]
        return out

    def _contact_summary(self) -> tuple[bool, int, float]:
        count = 0
        force_proxy = 0.0
        object_geoms = set(self.object_geom_ids)
        backbone_geoms = set(self.backbone_geom_ids)
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if (g1 in object_geoms and g2 in backbone_geoms) or (g2 in object_geoms and g1 in backbone_geoms):
                count += 1
                dist = float(getattr(c, 'dist', 0.0))
                force_proxy += max(0.0, -dist)
        return count > 0, count, force_proxy

    def _make_obs(self) -> Observation:
        object_pos = np.asarray(self.data.xpos[self.obj_body_id], dtype=np.float32).copy()
        cvel = np.asarray(self.data.cvel[self.obj_body_id], dtype=np.float32)
        object_vel = cvel[3:6].copy() if cvel.shape[0] >= 6 else np.zeros(3, dtype=np.float32)
        tip_pos = np.asarray(self.data.xpos[self.tip_body_id], dtype=np.float32).copy()
        body_positions = np.asarray([self.data.xpos[bid] for bid in self.body_ids], dtype=np.float32) if self.body_ids else np.zeros((0, 3), dtype=np.float32)
        actuator_force = np.asarray(getattr(self.data, 'actuator_force', np.zeros((len(self.act_ids),), dtype=np.float32)), dtype=np.float32).copy()
        tendon_length = np.asarray(getattr(self.data, 'ten_length', np.zeros((len(self.ten_ids),), dtype=np.float32)), dtype=np.float32).copy()
        tendon_velocity = np.asarray(getattr(self.data, 'ten_velocity', np.zeros((len(self.ten_ids),), dtype=np.float32)), dtype=np.float32).copy()
        contact, contact_count, contact_force_proxy = self._contact_summary()

        imu_acc = self._collect_sensor_block(self.acc_sensors, 3)
        imu_gyro = self._collect_sensor_block(self.gyro_sensors, 3)
        imu_quat = self._collect_sensor_block(self.quat_sensors, 4)

        return Observation(
            t=float(self.data.time),
            ctrl=np.asarray(self.data.ctrl[:self.model.nu], dtype=np.float32).copy(),
            actuator_force=actuator_force[:len(self.act_ids)],
            tendon_length=tendon_length[:len(self.ten_ids)],
            tendon_velocity=tendon_velocity[:len(self.ten_ids)],
            contact=contact,
            contact_count=contact_count,
            contact_force_proxy=float(contact_force_proxy),
            object_pos=object_pos,
            object_vel=object_vel,
            tip_pos=tip_pos,
            imu_acc=imu_acc,
            imu_gyro=imu_gyro,
            imu_quat=imu_quat,
            body_positions=body_positions,
        )

    def close(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        if self.tmp_xml.exists():
            try:
                os.remove(self.tmp_xml)
            except OSError:
                pass
