from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpiRobMultiTaskLoss(nn.Module):
    def __init__(
        self,
        distill_weight: float = 0.10,
        family_class_weights: list[float] | None = None,
        family_loss_weight: float = 4.0,
        quality_loss_weight: float = 1.0,
        size_loss_weight: float = 0.2,
        action_loss_weight: float = 0.5,
        trust_loss_weight: float = 0.2,
        distill_trust: bool = True,
    ) -> None:
        super().__init__()
        self.distill_weight = distill_weight
        self.family_loss_weight = family_loss_weight
        self.quality_loss_weight = quality_loss_weight
        self.size_loss_weight = size_loss_weight
        self.action_loss_weight = action_loss_weight
        self.trust_loss_weight = trust_loss_weight
        self.distill_trust = distill_trust

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

        if family_class_weights is not None:
            w = torch.tensor(family_class_weights, dtype=torch.float32)
            self.register_buffer("family_class_weights", w)
        else:
            self.family_class_weights = None

    def family_ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = None
        if self.family_class_weights is not None:
            weight = self.family_class_weights.to(logits.device)
        return F.cross_entropy(logits, targets, weight=weight)

    def forward(self, outputs: dict, batch: dict, teacher_outputs: dict | None = None) -> dict:
        success = self.bce(outputs["success_logit"], batch["success"])

        # smooth_l1 is more stable than plain MSE for the bimodal quality target.
        quality = F.smooth_l1_loss(outputs["quality"], batch["quality"], beta=0.1)

        family = self.family_ce(outputs["family_logits"], batch["family_id"])
        size = self.mse(outputs["size"], batch["size"])
        action = F.cross_entropy(outputs["action_logits"], batch["action_id"])
        trust = self.mse(outputs["trust"], batch["trust_target"])

        total = (
            success
            + self.quality_loss_weight * quality
            + self.family_loss_weight * family
            + self.size_loss_weight * size
            + self.action_loss_weight * action
            + self.trust_loss_weight * trust
        )

        distill = torch.tensor(0.0, device=total.device)
        if teacher_outputs is not None and self.distill_weight > 0.0:
            distill_terms = [
                F.mse_loss(outputs["success_prob"], teacher_outputs["success_prob"]),
                F.mse_loss(outputs["quality"], teacher_outputs["quality"]),
            ]
            if self.distill_trust:
                distill_terms.append(F.mse_loss(outputs["trust"], teacher_outputs["trust"]))
            distill = sum(distill_terms) / float(len(distill_terms))
            total = total + self.distill_weight * distill

        return {
            "success": success,
            "quality": quality,
            "family": family,
            "size": size,
            "action": action,
            "trust": trust,
            "distill": distill,
            "total": total,
        }



def fgsm_perturb(model, batch: dict, eps: float, criterion: SpiRobMultiTaskLoss) -> torch.Tensor:
    imu = batch["imu"].detach().clone().requires_grad_(True)
    outputs = model(imu, batch["mask"], batch["actuator"])
    loss = criterion(outputs, batch)["total"]
    loss.backward()
    grad = imu.grad.detach().sign() if imu.grad is not None else torch.zeros_like(imu)
    return (imu + eps * grad).detach()
