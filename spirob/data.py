from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .corruption import CorruptionConfig, corrupt_window


@dataclass
class NormalizationStats:
    imu_mean: np.ndarray
    imu_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray

    def to_jsonable(self) -> dict:
        return {
            "imu_mean": self.imu_mean.tolist(),
            "imu_std": self.imu_std.tolist(),
            "act_mean": self.act_mean.tolist(),
            "act_std": self.act_std.tolist(),
        }


class SpiRobWindowDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        stats: NormalizationStats | None = None,
        corrupt_train: bool = False,
        corruption_config: CorruptionConfig | None = None,
        seed: int = 7,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.stats = stats
        self.corrupt_train = corrupt_train
        self.corruption_config = corruption_config or CorruptionConfig()
        self.rng = random.Random(seed)

        self.items = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if split in {"all", "*"} or item.get("split") == split:
                    self.items.append(item)
        if not self.items:
            raise ValueError(f"No samples found for split={split!r} in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        arr = np.load(item["path"], allow_pickle=True)

        imu = arr["imu"].astype(np.float32)
        mask = arr["mask"].astype(np.float32)
        actuator = arr["actuator"].astype(np.float32)
        trust_target = arr["trust_target"].astype(np.float32)
        corruption_type = str(arr.get("corruption_type", "clean"))
        corruption_strength = float(np.asarray(arr.get("corruption_strength", 0.0)).item())

        if self.corrupt_train:
            imu, mask, trust_target, corruption_type, corruption_strength = corrupt_window(
                imu, mask, self.corruption_config, self.rng
            )

        if self.stats is not None:
            imu = (imu - self.stats.imu_mean.reshape(1, 1, -1)) / np.maximum(
                self.stats.imu_std.reshape(1, 1, -1), 1e-6
            )
            actuator = (actuator - self.stats.act_mean.reshape(1, -1)) / np.maximum(
                self.stats.act_std.reshape(1, -1), 1e-6
            )

        sample = {
            "imu": torch.from_numpy(imu),
            "mask": torch.from_numpy(mask),
            "actuator": torch.from_numpy(actuator),
            "success": torch.tensor(float(arr["success"]), dtype=torch.float32),
            "quality": torch.tensor(float(arr["quality"]), dtype=torch.float32),
            "family_id": torch.tensor(int(arr["family_id"]), dtype=torch.long),
            "size": torch.tensor(float(arr["size"]), dtype=torch.float32),
            "action_id": torch.tensor(int(arr["action_id"]), dtype=torch.long),
            "phase_id": torch.tensor(int(arr["phase_id"]), dtype=torch.long),
            "trust_target": torch.from_numpy(trust_target.astype(np.float32)),
            "corruption_type": corruption_type,
            "corruption_strength": torch.tensor(corruption_strength, dtype=torch.float32),
            "trial_id": torch.tensor(int(item.get("trial_id", -1)), dtype=torch.long),
            "window_id": torch.tensor(int(item.get("window_id", index)), dtype=torch.long),
            "path": item["path"],
            "split": item.get("split", self.split),
        }
        return sample



def collate_fn(batch: list[dict]) -> dict:
    out = {}
    for key in batch[0].keys():
        if torch.is_tensor(batch[0][key]):
            out[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            out[key] = [b[key] for b in batch]
    return out



def _iter_manifest_paths(manifest_path: str | Path, split: str = "train") -> Iterable[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if split in {"all", "*"} or item.get("split") == split:
                yield item["path"]



def compute_stats_from_manifest(manifest_path: str | Path, split: str = "train") -> NormalizationStats:
    imu_sum = np.zeros((6,), dtype=np.float64)
    imu_sq = np.zeros((6,), dtype=np.float64)
    act_sum = None
    act_sq = None
    imu_count = 0
    act_count = 0

    for path in _iter_manifest_paths(manifest_path, split=split):
        arr = np.load(path, allow_pickle=True)
        imu = arr["imu"].astype(np.float64)
        actuator = arr["actuator"].astype(np.float64)

        if act_sum is None:
            act_dim = actuator.shape[-1]
            act_sum = np.zeros((act_dim,), dtype=np.float64)
            act_sq = np.zeros((act_dim,), dtype=np.float64)

        imu_flat = imu.reshape(-1, imu.shape[-1])
        act_flat = actuator.reshape(-1, actuator.shape[-1])

        imu_sum += imu_flat.sum(axis=0)
        imu_sq += np.square(imu_flat).sum(axis=0)
        act_sum += act_flat.sum(axis=0)
        act_sq += np.square(act_flat).sum(axis=0)

        imu_count += imu_flat.shape[0]
        act_count += act_flat.shape[0]

    if imu_count == 0 or act_count == 0 or act_sum is None or act_sq is None:
        return NormalizationStats(
            imu_mean=np.zeros((6,), dtype=np.float32),
            imu_std=np.ones((6,), dtype=np.float32),
            act_mean=np.zeros((2,), dtype=np.float32),
            act_std=np.ones((2,), dtype=np.float32),
        )

    imu_mean = imu_sum / imu_count
    imu_var = np.maximum(imu_sq / imu_count - np.square(imu_mean), 1e-8)

    act_mean = act_sum / act_count
    act_var = np.maximum(act_sq / act_count - np.square(act_mean), 1e-8)

    return NormalizationStats(
        imu_mean=imu_mean.astype(np.float32),
        imu_std=np.sqrt(imu_var).astype(np.float32),
        act_mean=act_mean.astype(np.float32),
        act_std=np.sqrt(act_var).astype(np.float32),
    )



def save_stats(stats: NormalizationStats, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(stats.to_jsonable(), indent=2), encoding="utf-8")
    return path



def load_stats(path: str | Path) -> NormalizationStats:
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    return NormalizationStats(
        imu_mean=np.asarray(obj["imu_mean"], dtype=np.float32),
        imu_std=np.maximum(np.asarray(obj["imu_std"], dtype=np.float32), 1e-6),
        act_mean=np.asarray(obj["act_mean"], dtype=np.float32),
        act_std=np.maximum(np.asarray(obj["act_std"], dtype=np.float32), 1e-6),
    )
