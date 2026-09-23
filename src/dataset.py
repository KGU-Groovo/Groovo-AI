from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


def load_dca_npz_dataset(npz_path):
    """
    3단계에서 만든 synthetic npz 파일을 읽고,
    DCA v2 입력 형식에 맞는지 확인한다.

    현재 score는 실제 사람 채점이 아니라
    rule-based pseudo label이다.
    """
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"npz 파일을 찾을 수 없습니다: {npz_path}")

    required_keys = [
        "idol_seqs",
        "user_seqs",
        "scores",
        "joint_errors",
        "highlight_joints",
    ]

    raw = np.load(npz_path)

    for key in required_keys:
        if key not in raw:
            raise KeyError(f"npz 파일에 필요한 key가 없습니다: {key}")

    idol_seqs = raw["idol_seqs"].astype(np.float32)
    user_seqs = raw["user_seqs"].astype(np.float32)
    scores = raw["scores"].astype(np.float32)
    joint_errors = raw["joint_errors"].astype(np.float32)
    highlight_joints = raw["highlight_joints"].astype(np.int64)

    if idol_seqs.ndim != 4:
        raise ValueError(f"idol_seqs는 4차원이어야 합니다. 현재 shape: {idol_seqs.shape}")

    if idol_seqs.shape[1:] != (30, 33, 3):
        raise ValueError(
            f"idol_seqs shape는 (N, 30, 33, 3)이어야 합니다. 현재 shape: {idol_seqs.shape}"
        )

    if user_seqs.shape != idol_seqs.shape:
        raise ValueError(
            f"user_seqs shape는 idol_seqs와 같아야 합니다. idol={idol_seqs.shape}, user={user_seqs.shape}"
        )

    num_samples = idol_seqs.shape[0]

    if scores.shape != (num_samples,):
        raise ValueError(f"scores shape는 (N,)이어야 합니다. 현재 shape: {scores.shape}")

    if joint_errors.shape != (num_samples, 33):
        raise ValueError(
            f"joint_errors shape는 (N, 33)이어야 합니다. 현재 shape: {joint_errors.shape}"
        )

    if highlight_joints.shape != (num_samples, 3):
        raise ValueError(
            f"highlight_joints shape는 (N, 3)이어야 합니다. 현재 shape: {highlight_joints.shape}"
        )

    if num_samples < 30:
        print(f"[경고] sample 수가 {num_samples}개입니다. 테스트용 작은 데이터일 수 있습니다.")

    zero_ratio = float(np.mean(scores <= 0.0))
    full_ratio = float(np.mean(scores >= 100.0))

    if zero_ratio > 0.5:
        print("[경고] score가 0점 근처에 많이 몰려 있습니다.")

    if full_ratio > 0.5:
        print("[경고] score가 100점 근처에 많이 몰려 있습니다.")

    return {
        "idol_seqs": idol_seqs,
        "user_seqs": user_seqs,
        "scores": scores,
        "joint_errors": joint_errors,
        "highlight_joints": highlight_joints,
    }


def combine_dca_npz_datasets(npz_paths):
    """여러 곡의 NPZ를 합치고, 각 샘플의 곡 ID를 함께 반환한다."""
    npz_paths = [Path(path) for path in npz_paths]
    if not npz_paths:
        raise ValueError("합칠 NPZ 파일이 없습니다.")

    datasets = [load_dca_npz_dataset(path) for path in npz_paths]
    keys = ["idol_seqs", "user_seqs", "scores", "joint_errors", "highlight_joints"]
    data = {key: np.concatenate([dataset[key] for dataset in datasets], axis=0) for key in keys}
    group_ids = [
        path.stem
        for path, dataset in zip(npz_paths, datasets)
        for _ in range(len(dataset["scores"]))
    ]
    return data, group_ids


def create_group_split_indices(
    group_ids,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """같은 기준 곡의 모든 합성 샘플을 하나의 split에만 넣는다."""
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("split ratio 합은 1이어야 합니다.")

    unique_groups = np.array(sorted(set(group_ids)))
    if len(unique_groups) < 3:
        raise ValueError("곡 단위 train / valid / test 분할에는 최소 3곡이 필요합니다.")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    train_count = max(int(len(unique_groups) * train_ratio), 1)
    valid_count = max(int(len(unique_groups) * valid_ratio), 1)
    test_count = len(unique_groups) - train_count - valid_count
    if test_count < 1:
        train_count -= 1
        test_count = 1

    group_splits = {
        "train": set(unique_groups[:train_count]),
        "valid": set(unique_groups[train_count:train_count + valid_count]),
        "test": set(unique_groups[train_count + valid_count:]),
    }
    return {
        f"{name}_indices": [index for index, group_id in enumerate(group_ids) if group_id in groups]
        for name, groups in group_splits.items()
    }


class DanceDCADataset(Dataset):
    """
    DCA-Net 학습 전용 Dataset.

    하나의 sample은 idol_seq와 user_seq를 함께 반환한다.
    DCA-Net은 두 동작을 비교해서 차이를 학습하는 모델이기 때문이다.

    highlight_joints는 현재 참고 정보다.
    바로 학습 타깃으로 쓰기보다는, 어떤 관절 차이가 컸는지 확인하는 용도로 먼저 사용한다.
    """

    def __init__(self, npz_path=None, data_dict=None):
        if npz_path is None and data_dict is None:
            raise ValueError("npz_path 또는 data_dict 중 하나는 반드시 필요합니다.")

        if npz_path is not None and data_dict is not None:
            raise ValueError("npz_path와 data_dict를 동시에 넣지 마세요.")

        if data_dict is None:
            data_dict = load_dca_npz_dataset(npz_path)

        self.idol_seqs = data_dict["idol_seqs"]
        self.user_seqs = data_dict["user_seqs"]
        self.scores = data_dict["scores"]
        self.joint_errors = data_dict["joint_errors"]
        self.highlight_joints = data_dict["highlight_joints"]

    def __len__(self):
        return self.idol_seqs.shape[0]

    def __getitem__(self, idx):
        score_100 = float(self.scores[idx])
        score_norm = score_100 / 100.0

        return {
            "idol_seq": torch.tensor(self.idol_seqs[idx], dtype=torch.float32),
            "user_seq": torch.tensor(self.user_seqs[idx], dtype=torch.float32),
            "score": torch.tensor(score_100, dtype=torch.float32),
            "score_norm": torch.tensor(score_norm, dtype=torch.float32),
            "joint_errors": torch.tensor(self.joint_errors[idx], dtype=torch.float32),
            "highlight_joints": torch.tensor(self.highlight_joints[idx], dtype=torch.long),
            "sample_index": torch.tensor(idx, dtype=torch.long),
        }


def create_split_indices(
    num_samples,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """
    Dataset index를 train / valid / test로 나눈다.
    random_split을 쓰지 않는 이유는 나중에 같은 split을 다시 쓰기 위해서다.
    """
    ratio_sum = train_ratio + valid_ratio + test_ratio

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"split ratio 합은 1이어야 합니다. 현재 합: {ratio_sum}")

    if num_samples < 3:
        raise ValueError(
            f"train / valid / test에 최소 1개씩 넣으려면 sample이 최소 3개 필요합니다. 현재: {num_samples}"
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    train_count = int(num_samples * train_ratio)
    valid_count = int(num_samples * valid_ratio)

    train_count = max(train_count, 1)
    valid_count = max(valid_count, 1)
    test_count = num_samples - train_count - valid_count

    if test_count < 1:
        test_count = 1

        if train_count >= valid_count and train_count > 1:
            train_count -= 1
        elif valid_count > 1:
            valid_count -= 1
        else:
            raise ValueError("각 split에 최소 1개 이상 배정할 수 없습니다.")

    train_indices = indices[:train_count].tolist()
    valid_indices = indices[train_count:train_count + valid_count].tolist()
    test_indices = indices[train_count + valid_count:].tolist()

    if len(train_indices) + len(valid_indices) + len(test_indices) != num_samples:
        raise RuntimeError("split index 개수 합이 전체 sample 수와 다릅니다.")

    return {
        "train_indices": [int(x) for x in train_indices],
        "valid_indices": [int(x) for x in valid_indices],
        "test_indices": [int(x) for x in test_indices],
    }


def save_split_indices(split_indices, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(split_indices, f, ensure_ascii=False, indent=2)


def load_split_indices(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_dataloaders(
    dataset,
    split_indices,
    batch_size=32,
    num_workers=0,
    pin_memory=False,
):
    train_subset = Subset(dataset, split_indices["train_indices"])
    valid_subset = Subset(dataset, split_indices["valid_indices"])
    test_subset = Subset(dataset, split_indices["test_indices"])

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        valid_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }


def summarize_dataset(data_dict):
    print("===== Dataset Summary =====")

    for key, value in data_dict.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")

    scores = data_dict["scores"]

    print("---------------------------")
    print("num_samples:", len(scores))
    print("score min:", float(np.min(scores)))
    print("score mean:", float(np.mean(scores)))
    print("score max:", float(np.max(scores)))
    print("score zero ratio:", float(np.mean(scores <= 0.0)))
    print("score full ratio:", float(np.mean(scores >= 100.0)))
    print("---------------------------")
    print("주의: 현재 score는 실제 사람 채점이 아니라 rule-based pseudo label입니다.")


def check_batch_shapes(batch):
    print("===== Batch Shapes =====")

    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: type={type(value)}")

    idol_seq = batch["idol_seq"]
    user_seq = batch["user_seq"]
    score = batch["score"]
    score_norm = batch["score_norm"]
    joint_errors = batch["joint_errors"]
    highlight_joints = batch["highlight_joints"]

    batch_size = idol_seq.shape[0]

    assert idol_seq.ndim == 4, f"idol_seq는 4차원이어야 합니다. 현재: {idol_seq.shape}"
    assert idol_seq.shape[1:] == (30, 33, 3), f"idol_seq shape 오류: {idol_seq.shape}"
    assert user_seq.shape == idol_seq.shape, "user_seq shape는 idol_seq와 같아야 합니다."
    assert score.shape == (batch_size,), f"score shape 오류: {score.shape}"
    assert score_norm.shape == (batch_size,), f"score_norm shape 오류: {score_norm.shape}"
    assert joint_errors.shape == (batch_size, 33), f"joint_errors shape 오류: {joint_errors.shape}"
    assert highlight_joints.shape == (batch_size, 3), f"highlight_joints shape 오류: {highlight_joints.shape}"

    assert torch.all(score_norm >= 0.0), "score_norm에 0보다 작은 값이 있습니다."
    assert torch.all(score_norm <= 1.0), "score_norm에 1보다 큰 값이 있습니다."

    print("Batch shape check passed.")
