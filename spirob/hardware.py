from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


class JsonlImuStream:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __iter__(self) -> Iterator[dict]:
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


class SocketCanMotorInterface:
    def __init__(self, channel: str = 'can0', interface: str = 'socketcan') -> None:
        try:
            import can
        except Exception as exc:
            raise RuntimeError('python-can is required for SocketCanMotorInterface') from exc
        self._can = can
        self.bus = can.interface.Bus(channel=channel, interface=interface)

    @staticmethod
    def _float_to_uint(x, x_min, x_max, bits):
        x = max(x_min, min(x_max, float(x)))
        span = x_max - x_min
        return int((x - x_min) * ((2 ** bits - 1) / span))

    def _form_payload(self, position, velocity, torque, kp, kd):
        u_p = self._float_to_uint(position, -40.0, 40.0, 16)
        u_v = self._float_to_uint(velocity, -40.0, 40.0, 14)
        u_t = self._float_to_uint(torque, -40.0, 40.0, 16)
        u_kp = self._float_to_uint(kp, 0.0, 1023.0, 10)
        u_kd = self._float_to_uint(kd, 0.0, 51.0, 8)
        data = [0] * 8
        data[0] = u_p & 0xFF
        data[1] = (u_p >> 8) & 0xFF
        data[2] = u_v & 0xFF
        data[3] = ((u_v >> 8) & 0x3F) | ((u_kp & 0x03) << 6)
        data[4] = (u_kp >> 2) & 0xFF
        data[5] = u_kd & 0xFF
        data[6] = u_t & 0xFF
        data[7] = (u_t >> 8) & 0xFF
        return bytes(data)

    @staticmethod
    def _form_can_id(cmd: int, motor_id: int) -> int:
        return ((cmd & 0x3F) << 5) | (motor_id & 0x1F)

    def enable(self, motor_id: int) -> None:
        msg = self._can.Message(arbitration_id=self._form_can_id(2, motor_id), data=b'', is_extended_id=False)
        self.bus.send(msg)

    def disable(self, motor_id: int) -> None:
        msg = self._can.Message(arbitration_id=self._form_can_id(1, motor_id), data=b'', is_extended_id=False)
        self.bus.send(msg)

    def send_control(self, motor_id: int, p_des: float, v_des: float, kp: float, kd: float, tau: float) -> None:
        msg = self._can.Message(
            arbitration_id=self._form_can_id(4, motor_id),
            data=self._form_payload(p_des, v_des, tau, kp, kd),
            is_extended_id=False,
        )
        self.bus.send(msg)

    def close(self) -> None:
        if hasattr(self.bus, 'shutdown'):
            self.bus.shutdown()
