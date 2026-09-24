from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm


def compute_pairwise_ranking_loss(pred_score_norm, target_score_norm, margin=0.0):
    """정답 품질 순서와 반대인 예측 쌍에만 벌점을 준다."""
    target_diff = target_score_norm[:, None] - target_score_norm[None, :]
    mask = target_diff > 1e-6
    if not torch.any(mask):
        return pred_score_norm.new_zeros(())
    pred_diff = pred_score_norm[:, None] - pred_score_norm[None, :]
    return torch.relu(float(margin) - pred_diff[mask]).mean()


def move_batch_to_device(batch, device):
    """
    batch dict 안의 tensor들을 device로 이동한다.
    """
    moved = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value

    return moved


def compute_losses(
    outputs,
    batch,
    score_loss_fn,
    joint_loss_fn,
    joint_loss_weight=0.2,
    ranking_loss_weight=0.15,
):
    """
    score loss와 joint error loss를 함께 계산한다.

    현재 score는 실제 사람 채점이 아니라 rule-based pseudo label이다.
    따라서 이번 학습은 실제 인간 평가 모델이 아니라 prototype 학습이다.
    """
    pred_score_norm = outputs["score_norm"]
    target_score_norm = batch["score_norm"]

    pred_joint_errors = outputs["pred_joint_errors"]
    target_joint_errors = batch["joint_errors"]

    score_loss = score_loss_fn(pred_score_norm, target_score_norm)
    joint_loss = joint_loss_fn(pred_joint_errors, target_joint_errors)
    ranking_loss = compute_pairwise_ranking_loss(pred_score_norm, target_score_norm)

    total_loss = score_loss + joint_loss_weight * joint_loss + ranking_loss_weight * ranking_loss

    return {
        "total_loss": total_loss,
        "score_loss": score_loss,
        "joint_loss": joint_loss,
        "ranking_loss": ranking_loss,
    }


def compute_regression_metrics(pred_score_norm, target_score_norm):
    """
    score_norm 기준 MAE/RMSE를 계산한다.
    0~1 기준 metric과 0~100 기준 metric을 함께 반환한다.
    """
    pred = pred_score_norm.detach()
    target = target_score_norm.detach()

    error = pred - target

    mae = torch.mean(torch.abs(error))
    rmse = torch.sqrt(torch.mean(error ** 2) + 1e-12)

    return {
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "mae_100": float((mae * 100.0).item()),
        "rmse_100": float((rmse * 100.0).item()),
    }


def _init_metric_sums():
    return {
        "loss": 0.0,
        "score_loss": 0.0,
        "joint_loss": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
        "mae_100": 0.0,
        "rmse_100": 0.0,
    }


def _average_metric_sums(sums, total_samples):
    return {key: value / max(total_samples, 1) for key, value in sums.items()}


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    score_loss_fn,
    joint_loss_fn,
    joint_loss_weight=0.2,
    gradient_clip_norm=1.0,
):
    model.train()

    total_samples = 0
    sums = _init_metric_sums()

    progress = tqdm(dataloader, desc="train", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)
        batch_size = batch["idol_seq"].shape[0]

        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch["idol_seq"], batch["user_seq"])

        losses = compute_losses(
            outputs,
            batch,
            score_loss_fn,
            joint_loss_fn,
            joint_loss_weight=joint_loss_weight,
        )

        total_loss = losses["total_loss"]

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise RuntimeError("NaN 또는 Inf loss가 발생하여 학습을 중단합니다.")

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )

        optimizer.step()

        metrics = compute_regression_metrics(
            outputs["score_norm"],
            batch["score_norm"],
        )

        total_samples += batch_size
        sums["loss"] += float(losses["total_loss"].item()) * batch_size
        sums["score_loss"] += float(losses["score_loss"].item()) * batch_size
        sums["joint_loss"] += float(losses["joint_loss"].item()) * batch_size
        sums["mae"] += metrics["mae"] * batch_size
        sums["rmse"] += metrics["rmse"] * batch_size
        sums["mae_100"] += metrics["mae_100"] * batch_size
        sums["rmse_100"] += metrics["rmse_100"] * batch_size

        progress.set_postfix(loss=float(total_loss.item()))

    return _average_metric_sums(sums, total_samples)


def evaluate_one_epoch(
    model,
    dataloader,
    device,
    score_loss_fn,
    joint_loss_fn,
    joint_loss_weight=0.2,
):
    model.eval()

    total_samples = 0
    sums = _init_metric_sums()

    progress = tqdm(dataloader, desc="valid", leave=False)

    with torch.no_grad():
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            batch_size = batch["idol_seq"].shape[0]

            outputs = model(batch["idol_seq"], batch["user_seq"])

            losses = compute_losses(
                outputs,
                batch,
                score_loss_fn,
                joint_loss_fn,
                joint_loss_weight=joint_loss_weight,
            )

            total_loss = losses["total_loss"]

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                raise RuntimeError("평가 중 NaN 또는 Inf loss가 발생했습니다.")

            metrics = compute_regression_metrics(
                outputs["score_norm"],
                batch["score_norm"],
            )

            total_samples += batch_size
            sums["loss"] += float(losses["total_loss"].item()) * batch_size
            sums["score_loss"] += float(losses["score_loss"].item()) * batch_size
            sums["joint_loss"] += float(losses["joint_loss"].item()) * batch_size
            sums["mae"] += metrics["mae"] * batch_size
            sums["rmse"] += metrics["rmse"] * batch_size
            sums["mae_100"] += metrics["mae_100"] * batch_size
            sums["rmse_100"] += metrics["rmse_100"] * batch_size

            progress.set_postfix(loss=float(total_loss.item()))

    return _average_metric_sums(sums, total_samples)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_valid_loss,
    model_config,
    history_record,
    note,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_valid_loss": float(best_valid_loss),
        "model_config": model_config,
        "history_record": history_record,
        "note": note,
    }

    torch.save(checkpoint, path)


def train_model(model, dataloaders, device, config):
    """
    DCA-Net v2 학습 루프.

    valid_loss 기준으로 best checkpoint를 저장한다.
    """
    score_loss_fn = nn.SmoothL1Loss()
    joint_loss_fn = nn.SmoothL1Loss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    max_epochs = int(config.get("max_epochs", 30))
    patience = int(config.get("patience", 8))
    min_delta = float(config.get("min_delta", 1e-5))
    joint_loss_weight = float(config.get("joint_loss_weight", 0.2))
    gradient_clip_norm = float(config.get("gradient_clip_norm", 1.0))

    best_checkpoint_path = config["best_checkpoint_path"]
    last_checkpoint_path = config["last_checkpoint_path"]
    model_config = config.get("model_config", {})

    best_valid_loss = float("inf")
    best_epoch = -1
    patience_count = 0
    history_records = []

    print("현재 score는 실제 사람 채점이 아니라 rule-based pseudo label입니다.")
    print("valid_loss가 빠르게 낮아져도 실제 인간 평가 성능이 좋다는 뜻은 아닙니다.")

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            device,
            score_loss_fn,
            joint_loss_fn,
            joint_loss_weight=joint_loss_weight,
            gradient_clip_norm=gradient_clip_norm,
        )

        valid_metrics = evaluate_one_epoch(
            model,
            dataloaders["valid"],
            device,
            score_loss_fn,
            joint_loss_fn,
            joint_loss_weight=joint_loss_weight,
        )

        record = {
            "epoch": int(epoch),
            "train_loss": train_metrics["loss"],
            "train_score_loss": train_metrics["score_loss"],
            "train_joint_loss": train_metrics["joint_loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_mae_100": train_metrics["mae_100"],
            "train_rmse_100": train_metrics["rmse_100"],
            "valid_loss": valid_metrics["loss"],
            "valid_score_loss": valid_metrics["score_loss"],
            "valid_joint_loss": valid_metrics["joint_loss"],
            "valid_mae": valid_metrics["mae"],
            "valid_rmse": valid_metrics["rmse"],
            "valid_mae_100": valid_metrics["mae_100"],
            "valid_rmse_100": valid_metrics["rmse_100"],
        }

        history_records.append(record)

        improved = valid_metrics["loss"] < best_valid_loss - min_delta

        if improved:
            best_valid_loss = valid_metrics["loss"]
            best_epoch = epoch
            patience_count = 0

            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch,
                best_valid_loss,
                model_config,
                record,
                "Best DCA-Net v2 checkpoint by valid loss. Trained on rule-based pseudo labels.",
            )
        else:
            patience_count += 1

        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            epoch,
            best_valid_loss,
            model_config,
            record,
            "Last DCA-Net v2 checkpoint. Trained on rule-based pseudo labels.",
        )

        print(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"train_loss={record['train_loss']:.6f} | "
            f"valid_loss={record['valid_loss']:.6f} | "
            f"valid_MAE_100={record['valid_mae_100']:.3f} | "
            f"best_epoch={best_epoch}"
        )

        if patience_count >= patience:
            print(f"Early stopping: {patience} epoch 동안 valid_loss 개선이 없었습니다.")
            break

    best_info = {
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "best_checkpoint_path": str(best_checkpoint_path),
    }

    return history_records, best_info


def save_history(history_records, csv_path, json_path):
    if len(history_records) == 0:
        raise ValueError("저장할 history_records가 비어 있습니다.")

    csv_path = Path(csv_path)
    json_path = Path(json_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history_records)
    history_df.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(history_records, f, ensure_ascii=False, indent=2)

    print("history csv 저장:", csv_path)
    print("history json 저장:", json_path)

    return history_df


def load_best_model(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"best checkpoint를 찾을 수 없습니다: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return checkpoint
