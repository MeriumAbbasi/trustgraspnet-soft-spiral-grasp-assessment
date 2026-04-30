from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    imu_dim: int = 6
    act_dim: int = 8
    sensor_hidden: int = 24
    fusion_hidden: int = 64
    temporal_hidden: int = 96
    num_families: int = 5
    num_actions: int = 4
    dropout: float = 0.10


class _TemporalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.dropout(F.relu(self.bn1(self.conv1(y))))
        y = self.dropout(F.relu(self.bn2(self.conv2(y))))
        y = y.transpose(1, 2)
        y, _ = self.gru(y)
        return y


class SpiRobTinyModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.sensor_proj = nn.Sequential(
            nn.Linear(config.imu_dim, config.sensor_hidden),
            nn.ReLU(),
            nn.Linear(config.sensor_hidden, config.sensor_hidden),
            nn.ReLU(),
        )
        self.act_proj = nn.Sequential(
            nn.Linear(config.act_dim, config.fusion_hidden // 2),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden // 2, config.fusion_hidden // 2),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(config.sensor_hidden + config.fusion_hidden // 2, config.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.temporal = _TemporalEncoder(config.fusion_hidden, config.temporal_hidden, config.dropout)

        self.success_head = nn.Linear(config.temporal_hidden, 1)
        self.quality_head = nn.Linear(config.temporal_hidden, 1)
        self.family_head = nn.Linear(config.temporal_hidden, config.num_families)
        self.size_head = nn.Linear(config.temporal_hidden, 1)
        self.action_head = nn.Linear(config.temporal_hidden, config.num_actions)
        self.confidence_head = nn.Linear(config.temporal_hidden, 1)

        self.trust_head = nn.Sequential(
            nn.Linear(config.sensor_hidden, config.sensor_hidden),
            nn.ReLU(),
            nn.Linear(config.sensor_hidden, 1),
        )

    def _masked_sensor_pool(self, sensor_feat: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = mask.unsqueeze(-1)
        denom = torch.clamp(m.sum(dim=2), min=1.0)
        pooled_t = (sensor_feat * m).sum(dim=2) / denom
        sensor_avg = (sensor_feat * m).sum(dim=1) / torch.clamp(m.sum(dim=1), min=1.0)
        return pooled_t, sensor_avg

    def forward(self, imu: torch.Tensor, mask: torch.Tensor, actuator: torch.Tensor) -> dict:
        sensor_feat = self.sensor_proj(imu)
        pooled_t, sensor_avg = self._masked_sensor_pool(sensor_feat, mask)
        act_feat = self.act_proj(actuator)
        fused = self.fuse(torch.cat([pooled_t, act_feat], dim=-1))
        temporal = self.temporal(fused)
        global_feat = temporal.mean(dim=1)

        success_logit = self.success_head(global_feat).squeeze(-1)
        success_prob = torch.sigmoid(success_logit)
        quality = torch.sigmoid(self.quality_head(global_feat).squeeze(-1))
        family_logits = self.family_head(global_feat)
        size = self.size_head(global_feat).squeeze(-1)
        action_logits = self.action_head(global_feat)
        confidence = torch.sigmoid(self.confidence_head(global_feat).squeeze(-1))
        trust = torch.sigmoid(self.trust_head(sensor_avg).squeeze(-1))

        return {
            "success_logit": success_logit,
            "success_prob": success_prob,
            "quality": quality,
            "family_logits": family_logits,
            "size": size,
            "action_logits": action_logits,
            "confidence": confidence,
            "trust": trust,
        }


class SpiRobTeacherModel(SpiRobTinyModel):
    def __init__(self, act_dim: int = 8, num_families: int = 5) -> None:
        super().__init__(
            ModelConfig(
                act_dim=act_dim,
                num_families=num_families,
                sensor_hidden=48,
                fusion_hidden=128,
                temporal_hidden=160,
                dropout=0.15,
            )
        )