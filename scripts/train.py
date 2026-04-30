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

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from spirob.corruption import CorruptionConfig
from spirob.data import SpiRobWindowDataset, collate_fn, load_stats
from spirob.losses import SpiRobMultiTaskLoss, fgsm_perturb
from spirob.metrics import binary_metrics, multiclass_metrics, regression_metrics, trust_metrics
from spirob.model import ModelConfig, SpiRobTeacherModel, SpiRobTinyModel


def compute_family_class_weights(dataset: SpiRobWindowDataset, num_families: int) -> list[float]:
    counts = np.zeros(num_families, dtype=np.float64)
    for item in dataset.items:
        arr = np.load(item["path"], allow_pickle=True)
        fid = int(arr["family_id"])
        if 0 <= fid < num_families:
            counts[fid] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_families * counts)
    weights = weights / weights.mean()
    return weights.tolist()


@torch.no_grad()
def evaluate_epoch(model, loader, device):
    model.eval()
    success_labels = []
    success_probs = []
    quality_true = []
    quality_pred = []
    family_true = []
    family_logits = []
    size_true = []
    size_pred = []
    trust_true = []
    trust_pred = []
    action_true = []
    action_logits = []

    for batch in loader:
        imu = batch["imu"].to(device)
        mask = batch["mask"].to(device)
        actuator = batch["actuator"].to(device)
        out = model(imu, mask, actuator)

        success_labels.append(batch["success"].cpu().numpy())
        success_probs.append(out["success_prob"].cpu().numpy())
        quality_true.append(batch["quality"].cpu().numpy())
        quality_pred.append(out["quality"].cpu().numpy())
        family_true.append(batch["family_id"].cpu().numpy())
        family_logits.append(out["family_logits"].cpu().numpy())
        size_true.append(batch["size"].cpu().numpy())
        size_pred.append(out["size"].cpu().numpy())
        trust_true.append(batch["trust_target"].cpu().numpy())
        trust_pred.append(out["trust"].cpu().numpy())
        action_true.append(batch["action_id"].cpu().numpy())
        action_logits.append(out["action_logits"].cpu().numpy())

    success_labels = np.concatenate(success_labels)
    success_probs = np.concatenate(success_probs)
    quality_true = np.concatenate(quality_true)
    quality_pred = np.concatenate(quality_pred)
    family_true = np.concatenate(family_true)
    family_logits = np.concatenate(family_logits)
    size_true = np.concatenate(size_true)
    size_pred = np.concatenate(size_pred)
    trust_true = np.concatenate(trust_true)
    trust_pred = np.concatenate(trust_pred)
    action_true = np.concatenate(action_true)
    action_logits = np.concatenate(action_logits)

    metrics = {}
    metrics.update({f"success/{k}": v for k, v in binary_metrics(success_labels, success_probs).items()})
    metrics.update({f"quality/{k}": v for k, v in regression_metrics(quality_true, quality_pred).items()})
    metrics.update({f"family/{k}": v for k, v in multiclass_metrics(family_true, family_logits).items()})
    metrics.update({f"size/{k}": v for k, v in regression_metrics(size_true, size_pred).items()})
    metrics.update({f"trust/{k}": v for k, v in trust_metrics(trust_true, trust_pred).items()})
    metrics.update({f"action/{k}": v for k, v in multiclass_metrics(action_true, action_logits).items()})
    return metrics


def load_input_config(data_dir: Path) -> dict:
    path = data_dir / "input_config.json"
    if not path.exists():
        return {
            "imu_preset": "all",
            "imu_indices": None,
            "actuator_preset": "full",
            "actuator_columns": [
                "ctrl_1",
                "ctrl_2",
                "actuator_force_1",
                "actuator_force_2",
                "tendon_length_1",
                "tendon_length_2",
                "tendon_velocity_1",
                "tendon_velocity_2",
            ],
            "act_dim": 8,
            "num_families": 3,
            "family_label_mode": "coarse3",
            "coarse_shape_labels": ["elongated", "square", "round"],
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def build_model(args, act_dim: int, num_families: int):
    if args.model_type == "teacher":
        return SpiRobTeacherModel(act_dim=act_dim, num_families=num_families)

    config = ModelConfig(
        act_dim=act_dim,
        num_families=num_families,
        sensor_hidden=args.sensor_hidden,
        fusion_hidden=args.fusion_hidden,
        temporal_hidden=args.temporal_hidden,
        dropout=args.dropout,
    )
    return SpiRobTinyModel(config)



def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--teacher", type=str, default="")
    ap.add_argument("--adv-eps", type=float, default=0.0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--model-type", choices=["student", "teacher"], default="student")

    ap.add_argument("--sensor-hidden", type=int, default=24)
    ap.add_argument("--fusion-hidden", type=int, default=64)
    ap.add_argument("--temporal-hidden", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.10)

    # ablation-friendly loss and corruption controls
    ap.add_argument("--family-loss-weight", type=float, default=4.0)
    ap.add_argument("--quality-loss-weight", type=float, default=1.0)
    ap.add_argument("--size-loss-weight", type=float, default=0.2)
    ap.add_argument("--action-loss-weight", type=float, default=0.5)
    ap.add_argument("--trust-loss-weight", type=float, default=0.2)
    ap.add_argument("--distill-weight", type=float, default=0.10)
    ap.add_argument("--no-distill-trust", action="store_true")
    ap.add_argument("--train-corruption-p-apply", type=float, default=0.50)

    args = ap.parse_args()

    set_all_seeds(args.seed, deterministic=args.deterministic)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(data_dir / "norm_stats.json")
    manifest = data_dir / "manifest.jsonl"
    input_config = load_input_config(data_dir)

    act_dim = int(input_config.get("act_dim", len(input_config.get("actuator_columns", [])) or 8))
    num_families = int(input_config.get("num_families", 3))

    train_corruption_cfg = CorruptionConfig(p_apply=float(args.train_corruption_p_apply))

    train_ds = SpiRobWindowDataset(
        manifest,
        split="train",
        stats=stats,
        corrupt_train=args.train_corruption_p_apply > 0.0,
        corruption_config=train_corruption_cfg,
        seed=args.seed,
    )
    val_ds = SpiRobWindowDataset(
        manifest,
        split="val",
        stats=stats,
        corrupt_train=False,
        seed=args.seed,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = build_model(args, act_dim=act_dim, num_families=num_families).to(device)

    teacher = None
    if args.model_type == "student" and args.teacher:
        teacher_ckpt = torch.load(args.teacher, map_location=device)
        teacher_config_dict = teacher_ckpt.get("config", {})
        teacher_act_dim = int(teacher_config_dict.get("act_dim", act_dim))
        teacher_num_families = int(teacher_config_dict.get("num_families", num_families))

        if teacher_act_dim != act_dim:
            raise ValueError(
                f"Teacher checkpoint act_dim={teacher_act_dim}, but current dataset/model act_dim={act_dim}."
            )
        if teacher_num_families != num_families:
            raise ValueError(
                f"Teacher checkpoint num_families={teacher_num_families}, but current dataset/model num_families={num_families}."
            )

        if teacher_config_dict:
            teacher = SpiRobTinyModel(ModelConfig(**teacher_config_dict)).to(device)
        else:
            teacher = SpiRobTeacherModel(act_dim=act_dim, num_families=num_families).to(device)

        teacher.load_state_dict(teacher_ckpt["model"])
        teacher.eval()

    print("Computing family class weights from training set …")
    family_class_weights = compute_family_class_weights(train_ds, num_families)
    print(f"  class weights: {[f'{w:.3f}' for w in family_class_weights]}")

    criterion = SpiRobMultiTaskLoss(
        distill_weight=args.distill_weight,
        family_class_weights=family_class_weights,
        family_loss_weight=args.family_loss_weight,
        quality_loss_weight=args.quality_loss_weight,
        size_loss_weight=args.size_loss_weight,
        action_loss_weight=args.action_loss_weight,
        trust_loss_weight=args.trust_loss_weight,
        distill_trust=not args.no_distill_trust,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_metric = -float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"{args.model_type} epoch {epoch}/{args.epochs}")

        for batch in pbar:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            imu = batch["imu"]
            if args.adv_eps > 0.0:
                imu = fgsm_perturb(model, batch, args.adv_eps, criterion)

            outputs = model(imu, batch["mask"], batch["actuator"])

            teacher_outputs = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_outputs = teacher(batch["imu"], batch["mask"], batch["actuator"])

            loss_dict = criterion(outputs, batch, teacher_outputs=teacher_outputs)
            loss_dict["total"].backward()
            optimizer.step()

            running += float(loss_dict["total"].item())
            pbar.set_postfix(loss=float(loss_dict["total"].item()))

        val_metrics = evaluate_epoch(model, val_loader, device)
        score = (
            val_metrics.get("success/auroc", 0.0)
            + 0.5 * val_metrics.get("family/accuracy", 0.0)
            - 0.1 * val_metrics.get("quality/rmse", 0.0)
        )

        record = {
            "epoch": epoch,
            "seed": args.seed,
            "train_loss": running / max(1, len(train_loader)),
            "model_type": args.model_type,
            "train_corruption_p_apply": args.train_corruption_p_apply,
            "trust_loss_weight": args.trust_loss_weight,
            "distill_weight": args.distill_weight,
            "distill_trust": not args.no_distill_trust,
            **val_metrics,
        }
        history.append(record)

        with open(outdir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if score > best_metric:
            best_metric = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model.config.__dict__,
                    "input_config": input_config,
                    "best_metric": best_metric,
                    "model_type": args.model_type,
                    "seed": args.seed,
                    "train_corruption_p_apply": args.train_corruption_p_apply,
                    "trust_loss_weight": args.trust_loss_weight,
                    "distill_weight": args.distill_weight,
                    "distill_trust": not args.no_distill_trust,
                },
                outdir / "best.pt",
            )

    torch.save(
        {
            "model": model.state_dict(),
            "config": model.config.__dict__,
            "input_config": input_config,
            "model_type": args.model_type,
            "seed": args.seed,
            "train_corruption_p_apply": args.train_corruption_p_apply,
            "trust_loss_weight": args.trust_loss_weight,
            "distill_weight": args.distill_weight,
            "distill_trust": not args.no_distill_trust,
        },
        outdir / "last.pt",
    )


if __name__ == "__main__":
    main()
