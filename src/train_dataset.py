import argparse
from pathlib import Path

import torch

from dataset import DanceDCADataset, combine_dca_npz_datasets, create_dataloaders, create_group_split_indices, save_split_indices
from evaluate import evaluate_all_splits, save_evaluation_outputs, save_json
from model import create_dca_model
from train import load_best_model, save_history, train_model


def find_dataset_paths(dataset_dir):
    paths = sorted(Path(dataset_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"NPZ 파일이 없습니다: {dataset_dir}")
    return paths


def select_device(requested_device):
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Groovo 기준 안무 NPZ로 DCA 모델을 학습한다.")
    parser.add_argument("--dataset-dir", default="data/references")
    parser.add_argument("--output-dir", default="artifacts/training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, cpu")
    args = parser.parse_args()

    dataset_paths = find_dataset_paths(args.dataset_dir)
    data, group_ids = combine_dca_npz_datasets(dataset_paths)
    split_indices = create_group_split_indices(group_ids, seed=args.seed)
    dataset = DanceDCADataset(data_dict=data)
    device = select_device(args.device)
    dataloaders = create_dataloaders(
        dataset,
        split_indices,
        batch_size=args.batch_size,
        pin_memory=device.type == "cuda",
    )

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    save_split_indices(split_indices, output_dir / "split_indices.json")
    split_groups = {
        split_name: sorted({group_ids[index] for index in indices})
        for split_name, indices in split_indices.items()
    }
    save_json(output_dir / "split_groups.json", split_groups)

    model = create_dca_model().to(device)
    config = {
        "max_epochs": args.epochs,
        "best_checkpoint_path": checkpoints_dir / "best.pt",
        "last_checkpoint_path": checkpoints_dir / "last.pt",
        "model_config": {},
    }
    print(f"device: {device}")
    print(f"datasets: {[path.name for path in dataset_paths]}")
    print(f"split groups: {split_groups}")
    history, best_info = train_model(model, dataloaders, device, config)
    save_history(history, output_dir / "history.csv", output_dir / "history.json")

    load_best_model(model, best_info["best_checkpoint_path"], device)
    summary, predictions, _ = evaluate_all_splits(model, dataloaders, device)
    leakage_report = {
        "leakage_detected": False,
        "grouping": "song",
        "split_groups": split_groups,
        "note": "Each song is assigned to exactly one split.",
    }
    save_evaluation_outputs(
        summary,
        predictions,
        leakage_report,
        output_dir / "evaluation_summary.json",
        output_dir / "evaluation_predictions.csv",
        output_dir / "leakage_report.json",
    )
    save_json(output_dir / "run_summary.json", {
        "datasets": [str(path) for path in dataset_paths],
        "device": str(device),
        "num_samples": len(dataset),
        "best": best_info,
        "note": "Scores are rule-based pseudo labels, not human ratings.",
    })


if __name__ == "__main__":
    main()
