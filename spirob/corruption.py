from __future__ import annotations

from dataclasses import dataclass
import random
import numpy as np


@dataclass
class CorruptionConfig:
    p_apply: float = 0.50
    noise_std: float = 0.03
    drift_std: float = 0.002
    dropout_prob: float = 0.20
    clip_level: float = 3.0
    max_delay_steps: int = 6
    freeze_prob: float = 0.15
    spoof_prob: float = 0.10


CORRUPTION_TYPES = ('clean', 'noise', 'drift', 'dropout', 'clip', 'delay', 'freeze', 'spoof')


def _sensor_subset(rng: random.Random, n_imus: int, frac_lo: float = 0.15, frac_hi: float = 0.45) -> list[int]:
    if n_imus <= 0:
        return []
    k = max(1, min(n_imus, int(round(rng.uniform(frac_lo, frac_hi) * n_imus))))
    return sorted(rng.sample(list(range(n_imus)), k))


def corrupt_window(
    imu: np.ndarray,
    mask: np.ndarray,
    config: CorruptionConfig,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    imu = np.asarray(imu, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=np.float32).copy()
    T, M, C = imu.shape
    trust_target = np.ones((M,), dtype=np.float32)

    if rng.random() > config.p_apply:
        return imu, mask, trust_target, 'clean', 0.0

    corr = rng.choice(CORRUPTION_TYPES[1:])
    idx = _sensor_subset(rng, M)
    strength = rng.uniform(0.25, 1.0)

    if corr == 'noise':
        std = config.noise_std * strength
        imu[:, idx, :] += np.random.normal(0.0, std, size=(T, len(idx), C)).astype(np.float32)
        trust_target[idx] = np.clip(1.0 - 0.5 * strength, 0.0, 1.0)
    elif corr == 'drift':
        step = np.random.normal(0.0, config.drift_std * strength, size=(T, len(idx), C)).astype(np.float32)
        imu[:, idx, :] += np.cumsum(step, axis=0)
        trust_target[idx] = np.clip(1.0 - 0.6 * strength, 0.0, 1.0)
    elif corr == 'dropout':
        for m in idx:
            if rng.random() < config.dropout_prob + 0.5 * strength:
                imu[:, m, :] = 0.0
                mask[:, m] = 0.0
                trust_target[m] = 0.0
    elif corr == 'clip':
        level = config.clip_level * max(0.2, 1.0 - 0.6 * strength)
        imu[:, idx, :] = np.clip(imu[:, idx, :], -level, level)
        trust_target[idx] = np.clip(1.0 - 0.4 * strength, 0.0, 1.0)
    elif corr == 'delay':
        delay = max(1, int(round(strength * config.max_delay_steps)))
        imu[delay:, idx, :] = imu[:-delay, idx, :]
        imu[:delay, idx, :] = imu[:1, idx, :]
        trust_target[idx] = np.clip(1.0 - 0.5 * strength, 0.0, 1.0)
    elif corr == 'freeze':
        freeze_at = rng.randrange(max(1, T // 4), max(2, T))
        imu[freeze_at:, idx, :] = imu[freeze_at:freeze_at + 1, idx, :]
        trust_target[idx] = np.clip(1.0 - 0.7 * strength, 0.0, 1.0)
    elif corr == 'spoof':
        source = rng.choice([m for m in range(M) if m not in idx]) if len(idx) < M else idx[0]
        for m in idx:
            imu[:, m, :] = imu[:, source, :]
            trust_target[m] = np.clip(0.2 - 0.1 * strength, 0.0, 1.0)
    else:
        corr = 'clean'
        strength = 0.0

    return imu, mask, trust_target, corr, float(strength)
