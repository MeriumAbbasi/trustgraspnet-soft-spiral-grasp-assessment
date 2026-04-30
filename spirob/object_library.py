from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Iterable, Sequence


@dataclass
class ObjectSpec:
    family: str
    size: float
    mass: float
    representative_diameter: float
    pose_x: float
    pose_z: float
    yaw: float = 0.0
    extras: dict = field(default_factory=dict)


GOAL_XZ = [
    (0.1931, 0.6790), (-0.0746, 0.7827), (0.2229, 0.5106), (0.1963, 0.6327),
    (0.0858, 0.7581), (-0.1978, 0.6585), (-0.2150, 0.5573), (-0.1964, 0.6079),
    (-0.0821, 0.6885), (-0.1767, 0.7049), (-0.0675, 0.7182), (0.2055, 0.7085),
    (0.1574, 0.6982), (0.1945, 0.5649), (-0.1408, 0.6714), (0.1708, 0.6500),
    (0.0812, 0.7177), (-0.0949, 0.7324), (0.1082, 0.7060), (0.1719, 0.6099),
    (0.0645, 0.6948), (0.2235, 0.6804), (-0.1637, 0.7413), (-0.1308, 0.7385),
    (-0.2097, 0.7142), (0.1110, 0.7756), (-0.1649, 0.6549), (0.2206, 0.5779),
    (-0.2270, 0.6230), (0.1546, 0.7459), (0.1337, 0.7242), (0.2255, 0.6399),
    (-0.2084, 0.5150), (-0.1224, 0.7653), (0.2271, 0.6085), (0.1418, 0.6426),
    (-0.1123, 0.6749), (-0.2275, 0.6847), (0.0712, 0.7902), (0.1993, 0.6012),
    (-0.1106, 0.7044), (0.0941, 0.6776),
]

FAMILIES = (
    'sphere',
    'cylinder',
    'capsule',
    'box',
    'flat',
    'branched',
    'irregular',
)


def workspace_positions() -> list[tuple[float, float]]:
    return list(GOAL_XZ)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _mass_from_size(size: float, density_scale: float = 1.0) -> float:
    # crude but stable heuristic for synthetic sampling
    return float(_clamp(0.020 + density_scale * 2500.0 * size ** 3, 0.015, 0.250))


def _make_sphere(size: float, pose_x: float, pose_z: float, yaw: float) -> ObjectSpec:
    radius = size
    return ObjectSpec(
        family='sphere',
        size=size,
        mass=_mass_from_size(size, 0.60),
        representative_diameter=2.0 * radius,
        pose_x=pose_x,
        pose_z=pose_z,
        yaw=yaw,
        extras={'radius': radius},
    )


def _make_cylinder(size: float, pose_x: float, pose_z: float, yaw: float) -> ObjectSpec:
    radius = size
    half_length = 1.3 * size
    return ObjectSpec(
        family='cylinder',
        size=size,
        mass=_mass_from_size(size, 0.70),
        representative_diameter=2.0 * radius,
        pose_x=pose_x,
        pose_z=pose_z,
        yaw=yaw,
        extras={'radius': radius, 'half_length': half_length},
    )


def _make_capsule(size: float, pose_x: float, pose_z: float, yaw: float) -> ObjectSpec:
    radius = 0.85 * size
    half_length = 1.6 * size
    return ObjectSpec(
        family='capsule',
        size=size,
        mass=_mass_from_size(size, 0.65),
        representative_diameter=2.0 * radius,
        pose_x=pose_x,
        pose_z=pose_z,
        yaw=yaw,
        extras={'radius': radius, 'half_length': half_length},
    )


def _make_box(size: float, pose_x: float, pose_z: float, yaw: float) -> ObjectSpec:
    hx = 1.2 * size
    hy = 0.9 * size
    hz = 1.1 * size
    return ObjectSpec(
        family='box',
        size=size,
        mass=_mass_from_size(size, 0.90),
        representative_diameter=2.0 * max(hx, hy, hz),
        pose_x=pose_x,
        pose_z=pose_z,
        yaw=yaw,
        extras={'half_sizes': [hx, hy, hz]},
    )


def _make_flat(size: float, pose_x: float, pose_z: float, yaw: float) -> ObjectSpec:
    hx = 1.6 * size
    hy = 0.8 * size
    hz = 0.35 * size
    return ObjectSpec(
        family='flat',
        size=size,
        mass=_mass_from_size(size, 0.45),
        representative_diameter=2.0 * max(hx, hy),
        pose_x=pose_x,
        pose_z=pose_z,
        yaw=yaw,
        extras={'half_sizes': [hx, hy, hz]},
    )


def _make_branched(size: float, pose_x: float, pose_z: float, yaw: float) -> ObjectSpec:
    trunk_r = 0.55 * size
    trunk_half = 1.3 * size
    arm_r = 0.32 * size
    arm_half = 0.9 * size
    return ObjectSpec(
        family='branched',
        size=size,
        mass=_mass_from_size(size, 0.55),
        representative_diameter=3.0 * size,
        pose_x=pose_x,
        pose_z=pose_z,
        yaw=yaw,
        extras={
            'trunk_radius': trunk_r,
            'trunk_half_length': trunk_half,
            'arm_radius': arm_r,
            'arm_half_length': arm_half,
        },
    )


def _make_irregular(size: float, pose_x: float, pose_z: float, yaw: float, rng: random.Random) -> ObjectSpec:
    blobs = []
    for _ in range(3):
        ang = rng.uniform(0.0, 2.0 * math.pi)
        rad = rng.uniform(0.15 * size, 0.55 * size)
        px = rad * math.cos(ang)
        pz = rad * math.sin(ang)
        blobs.append({'pos': [px, 0.0, pz], 'radius': rng.uniform(0.35 * size, 0.75 * size)})
    return ObjectSpec(
        family='irregular',
        size=size,
        mass=_mass_from_size(size, 0.60),
        representative_diameter=2.8 * size,
        pose_x=pose_x,
        pose_z=pose_z,
        yaw=yaw,
        extras={'blobs': blobs},
    )


def sample_object(rng: random.Random, positions: Sequence[tuple[float, float]] | None = None) -> ObjectSpec:
    if positions is None:
        positions = GOAL_XZ
    pose_x, pose_z = rng.choice(list(positions))
    family = rng.choice(FAMILIES)
    size = rng.uniform(0.010, 0.023)
    yaw = rng.uniform(-0.6, 0.6)

    if family == 'sphere':
        return _make_sphere(size, pose_x, pose_z, yaw)
    if family == 'cylinder':
        return _make_cylinder(size, pose_x, pose_z, yaw)
    if family == 'capsule':
        return _make_capsule(size, pose_x, pose_z, yaw)
    if family == 'box':
        return _make_box(size, pose_x, pose_z, yaw)
    if family == 'flat':
        return _make_flat(size, pose_x, pose_z, yaw)
    if family == 'branched':
        return _make_branched(size, pose_x, pose_z, yaw)
    return _make_irregular(size, pose_x, pose_z, yaw, rng)
