from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def load_json(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, ensure_ascii=False, indent=2)


def move_batch_to_device(batch, device):
    moved = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value

    return moved


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def compute_score_metrics(pred_score_norm, target_score_norm):
    pred_norm = _to_numpy(pred_score_norm).astype(np.float64)
    target_norm = _to_numpy(target_score_norm).astype(np.float64)

    pred_100 = pred_norm * 100.0
    target_100 = target_norm * 100.0

    error_norm = pred_norm - target_norm
    error_100 = pred_100 - target_100

    mae = float(np.mean(np.abs(error_norm)))
    rmse = float(np.sqrt(np.mean(error_norm ** 2)))
    mae_100 = float(np.mean(np.abs(error_100)))
    rmse_100 = float(np.sqrt(np.mean(error_100 ** 2)))

    pred_std = float(np.std(pred_100))
    target_std = float(np.std(target_100))

    if pred_std < 1e-12 or target_std < 1e-12:
        correlation = None
    else:
        corr = float(np.corrcoef(pred_100, target_100)[0, 1])
        if np.isnan(corr) or np.isinf(corr):
            correlation = None
        else:
            correlation = corr

    mean_prediction_warning = False
    if target_std > 0.0 and pred_std < target_std * 0.3:
        mean_prediction_warning = True

    return {
        "mae": mae,
        "rmse": rmse,
        "mae_100": mae_100,
        "rmse_100": rmse_100,
        "correlation": correlation,
        "pred_score_100_mean": float(np.mean(pred_100)),
        "pred_score_100_std": pred_std,
        "pred_score_100_min": float(np.min(pred_100)),
        "pred_score_100_max": float(np.max(pred_100)),
        "pred_score_100_range": float(np.max(pred_100) - np.min(pred_100)),
        "target_score_100_mean": float(np.mean(target_100)),
        "target_score_100_std": target_std,
        "target_score_100_min": float(np.min(target_100)),
        "target_score_100_max": float(np.max(target_100)),
        "target_score_100_range": float(np.max(target_100) - np.min(target_100)),
        "mean_prediction_warning": mean_prediction_warning,
    }


def compute_joint_metrics(pred_joint_errors, target_joint_errors):
    pred = _to_numpy(pred_joint_errors).astype(np.float64)
    target = _to_numpy(target_joint_errors).astype(np.float64)

    abs_error = np.abs(pred - target)

    joint_mae = float(np.mean(abs_error))
    joint_mae_by_index = np.mean(abs_error, axis=0)

    top5_indices = np.argsort(joint_mae_by_index)[::-1][:5]

    top5 = []
    for idx in top5_indices:
        top5.append(
            {
                "joint_index": int(idx),
                "mae": float(joint_mae_by_index[idx]),
            }
        )

    return {
        "joint_mae": joint_mae,
        "joint_mae_by_index": joint_mae_by_index,
        "top5_high_error_joints": top5,
    }


def evaluate_split(model, dataloader, device, split_name):
    model.eval()

    rows = []
    target_joint_errors_list = []
    pred_joint_errors_list = []

    progress = tqdm(dataloader, desc=f"evaluate {split_name}", leave=False)

    with torch.no_grad():
        for batch in progress:
            batch = move_batch_to_device(batch, device)

            outputs = model(batch["idol_seq"], batch["user_seq"])

            sample_indices = batch["sample_index"].detach().cpu().numpy()
            target_score_norm = batch["score_norm"].detach().cpu().numpy()
            pred_score_norm = outputs["score_norm"].detach().cpu().numpy()

            target_joint_errors = batch["joint_errors"].detach().cpu().numpy()
            pred_joint_errors = outputs["pred_joint_errors"].detach().cpu().numpy()
            highlight_joints = outputs["highlight_joints"].detach().cpu().numpy()

            target_joint_errors_list.append(target_joint_errors)
            pred_joint_errors_list.append(pred_joint_errors)

            for i in range(len(sample_indices)):
                target_100 = float(target_score_norm[i] * 100.0)
                pred_100 = float(pred_score_norm[i] * 100.0)

                highlight_list = [int(x) for x in highlight_joints[i].tolist()]
                highlight_str = ",".join(str(x) for x in highlight_list)

                rows.append(
                    {
                        "split": split_name,
                        "sample_index": int(sample_indices[i]),
                        "target_score_norm": float(target_score_norm[i]),
                        "pred_score_norm": float(pred_score_norm[i]),
                        "target_score_100": target_100,
                        "pred_score_100": pred_100,
                        "abs_error_100": float(abs(pred_100 - target_100)),
                        "highlight_joints_pred": highlight_str,
                    }
                )

    predictions_df = pd.DataFrame(rows)

    return {
        "split_name": split_name,
        "predictions_df": predictions_df,
        "target_joint_errors": np.concatenate(target_joint_errors_list, axis=0),
        "pred_joint_errors": np.concatenate(pred_joint_errors_list, axis=0),
    }


def evaluate_all_splits(model, dataloaders, device):
    split_results = {}

    for split_name in ["train", "valid", "test"]:
        split_results[split_name] = evaluate_split(
            model=model,
            dataloader=dataloaders[split_name],
            device=device,
            split_name=split_name,
        )

    predictions_df = pd.concat(
        [
            split_results["train"]["predictions_df"],
            split_results["valid"]["predictions_df"],
            split_results["test"]["predictions_df"],
        ],
        ignore_index=True,
    )

    summary = {
        "note": "Evaluation uses rule-based pseudo labels, not real human scores.",
        "splits": {},
    }

    for split_name in ["train", "valid", "test"]:
        split_df = split_results[split_name]["predictions_df"]

        score_metrics = compute_score_metrics(
            split_df["pred_score_norm"].to_numpy(),
            split_df["target_score_norm"].to_numpy(),
        )

        joint_metrics = compute_joint_metrics(
            split_results[split_name]["pred_joint_errors"],
            split_results[split_name]["target_joint_errors"],
        )

        summary["splits"][split_name] = {
            "num_samples": int(len(split_df)),
            "score_metrics": score_metrics,
            "joint_metrics": joint_metrics,
        }

    all_score_metrics = compute_score_metrics(
        predictions_df["pred_score_norm"].to_numpy(),
        predictions_df["target_score_norm"].to_numpy(),
    )

    all_pred_joint = np.concatenate(
        [
            split_results["train"]["pred_joint_errors"],
            split_results["valid"]["pred_joint_errors"],
            split_results["test"]["pred_joint_errors"],
        ],
        axis=0,
    )

    all_target_joint = np.concatenate(
        [
            split_results["train"]["target_joint_errors"],
            split_results["valid"]["target_joint_errors"],
            split_results["test"]["target_joint_errors"],
        ],
        axis=0,
    )

    summary["all"] = {
        "num_samples": int(len(predictions_df)),
        "score_metrics": all_score_metrics,
        "joint_metrics": compute_joint_metrics(all_pred_joint, all_target_joint),
    }

    return summary, predictions_df, split_results


def diagnose_mean_prediction(predictions_df):
    report = {}

    for split_name, split_df in predictions_df.groupby("split"):
        target_std = float(split_df["target_score_100"].std(ddof=0))
        pred_std = float(split_df["pred_score_100"].std(ddof=0))

        warning = False

        if target_std > 0.0 and pred_std < target_std * 0.3:
            warning = True

        report[split_name] = {
            "target_score_100_mean": float(split_df["target_score_100"].mean()),
            "target_score_100_std": target_std,
            "target_score_100_min": float(split_df["target_score_100"].min()),
            "target_score_100_max": float(split_df["target_score_100"].max()),
            "target_score_100_range": float(split_df["target_score_100"].max() - split_df["target_score_100"].min()),
            "pred_score_100_mean": float(split_df["pred_score_100"].mean()),
            "pred_score_100_std": pred_std,
            "pred_score_100_min": float(split_df["pred_score_100"].min()),
            "pred_score_100_max": float(split_df["pred_score_100"].max()),
            "pred_score_100_range": float(split_df["pred_score_100"].max() - split_df["pred_score_100"].min()),
            "mean_prediction_warning": warning,
        }

    return report


def _metadata_to_window_map(metadata):
    window_map = {}

    if isinstance(metadata, dict):
        if "samples" in metadata and isinstance(metadata["samples"], list):
            metadata_items = metadata["samples"]
        elif "metadata" in metadata and isinstance(metadata["metadata"], list):
            metadata_items = metadata["metadata"]
        else:
            metadata_items = []
    elif isinstance(metadata, list):
        metadata_items = metadata
    else:
        metadata_items = []

    for position, item in enumerate(metadata_items):
        if not isinstance(item, dict):
            continue

        sample_index = item.get(
            "sample_index",
            item.get("sample_id", item.get("index", position)),
        )

        window_index = item.get(
            "window_index",
            item.get("source_window_index", None),
        )

        if window_index is None:
            continue

        try:
            sample_index = int(sample_index)
        except Exception:
            sample_index = position

        window_map[sample_index] = int(window_index)

    return window_map


def check_window_leakage(split_indices, metadata):
    window_map = _metadata_to_window_map(metadata)

    split_window_sets = {}

    split_key_pairs = [
        ("train", "train_indices"),
        ("valid", "valid_indices"),
        ("test", "test_indices"),
    ]

    for split_name, key in split_key_pairs:
        window_set = set()
        missing_sample_count = 0

        for sample_index in split_indices[key]:
            sample_index = int(sample_index)

            if sample_index in window_map:
                window_set.add(window_map[sample_index])
            else:
                missing_sample_count += 1

        split_window_sets[split_name] = {
            "window_set": window_set,
            "missing_sample_count": missing_sample_count,
        }

    train_windows = split_window_sets["train"]["window_set"]
    valid_windows = split_window_sets["valid"]["window_set"]
    test_windows = split_window_sets["test"]["window_set"]

    train_valid_overlap = train_windows & valid_windows
    train_test_overlap = train_windows & test_windows
    valid_test_overlap = valid_windows & test_windows

    total_unique_windows = len(train_windows | valid_windows | test_windows)
    leakage_windows = train_valid_overlap | train_test_overlap | valid_test_overlap
    leakage_count = len(leakage_windows)

    if total_unique_windows == 0:
        leakage_ratio = None
    else:
        leakage_ratio = leakage_count / total_unique_windows

    leakage_detected = leakage_count > 0

    return {
        "metadata_window_map_size": int(len(window_map)),
        "train_window_count": int(len(train_windows)),
        "valid_window_count": int(len(valid_windows)),
        "test_window_count": int(len(test_windows)),
        "train_missing_sample_count": int(split_window_sets["train"]["missing_sample_count"]),
        "valid_missing_sample_count": int(split_window_sets["valid"]["missing_sample_count"]),
        "test_missing_sample_count": int(split_window_sets["test"]["missing_sample_count"]),
        "total_unique_window_count": int(total_unique_windows),
        "train_valid_overlap_count": int(len(train_valid_overlap)),
        "train_test_overlap_count": int(len(train_test_overlap)),
        "valid_test_overlap_count": int(len(valid_test_overlap)),
        "train_valid_overlap_windows": sorted([int(x) for x in train_valid_overlap]),
        "train_test_overlap_windows": sorted([int(x) for x in train_test_overlap]),
        "valid_test_overlap_windows": sorted([int(x) for x in valid_test_overlap]),
        "leakage_count": int(leakage_count),
        "leakage_ratio": leakage_ratio,
        "leakage_detected": bool(leakage_detected),
        "note": "If leakage_detected is true, sample-level split can overestimate performance.",
    }


def save_evaluation_outputs(
    summary,
    predictions_df,
    leakage_report,
    summary_path,
    predictions_path,
    leakage_path,
):
    summary_path = Path(summary_path)
    predictions_path = Path(predictions_path)
    leakage_path = Path(leakage_path)

    save_json(summary_path, summary)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_path, index=False)

    save_json(leakage_path, leakage_report)

    print("evaluation summary 저장:", summary_path)
    print("evaluation predictions csv 저장:", predictions_path)
    print("leakage report 저장:", leakage_path)
