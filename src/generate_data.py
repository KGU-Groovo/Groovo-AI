from pathlib import Path
import json
import numpy as np

from normalization import compute_joint_errors, get_topk_highlight_joints


def create_sliding_windows(sequence, frame_length=30, stride=15):
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.ndim != 3:
        raise ValueError(f"sequence must have shape (N, 33, 3), got {sequence.shape}")

    if sequence.shape[1:] != (33, 3):
        raise ValueError(
            f"sequence shape after N must be (33, 3), got {sequence.shape[1:]}"
        )

    if sequence.shape[0] < frame_length:
        raise ValueError(f"N must be >= {frame_length}, got {sequence.shape[0]}")

    windows = []

    for start in range(0, sequence.shape[0] - frame_length + 1, stride):
        end = start + frame_length
        windows.append(sequence[start:end])

    return np.stack(windows, axis=0).astype(np.float32)


def perturb_sequence(idol_seq, error_type, rng):
    idol_seq = np.asarray(idol_seq, dtype=np.float32)
    user_seq = idol_seq.copy()

    if user_seq.shape != (30, 33, 3):
        raise ValueError(f"idol_seq must have shape (30, 33, 3), got {user_seq.shape}")

    if error_type == "clean":
        user_seq += rng.normal(0.0, 0.005, size=user_seq.shape).astype(np.float32)

    elif error_type == "noise_small":
        user_seq += rng.normal(0.0, 0.025, size=user_seq.shape).astype(np.float32)

    elif error_type == "noise_large":
        user_seq += rng.normal(0.0, 0.080, size=user_seq.shape).astype(np.float32)

    elif error_type == "timing_delay":
        delay = int(rng.integers(2, 6))
        user_seq = np.roll(user_seq, shift=delay, axis=0)
        user_seq[:delay] = idol_seq[0]
        user_seq += rng.normal(0.0, 0.015, size=user_seq.shape).astype(np.float32)

    elif error_type == "left_arm_low":
        joints = [11, 13, 15]
        offset = float(rng.uniform(0.12, 0.25))
        user_seq[:, joints, 1] += offset
        user_seq[:, joints, 0] += rng.normal(
            0.0,
            0.025,
            size=(user_seq.shape[0], len(joints)),
        ).astype(np.float32)

    elif error_type == "right_arm_low":
        joints = [12, 14, 16]
        offset = float(rng.uniform(0.12, 0.25))
        user_seq[:, joints, 1] += offset
        user_seq[:, joints, 0] += rng.normal(
            0.0,
            0.025,
            size=(user_seq.shape[0], len(joints)),
        ).astype(np.float32)

    elif error_type == "leg_shift":
        joints = [23, 24, 25, 26, 27, 28]
        shift = rng.normal(
            0.0,
            0.09,
            size=(1, len(joints), 3),
        ).astype(np.float32)
        user_seq[:, joints, :] += shift

    elif error_type == "upper_body_shift":
        joints = [0, 11, 12, 13, 14, 15, 16, 23, 24]
        shift = np.array(
            [
                float(rng.uniform(-0.18, 0.18)),
                float(rng.uniform(-0.08, 0.08)),
                0.0,
            ],
            dtype=np.float32,
        )
        user_seq[:, joints, :] += shift

    else:
        raise ValueError(f"Unsupported error_type: {error_type}")

    return user_seq.astype(np.float32)


def compute_score_from_error(joint_errors, error_type):
    joint_errors = np.asarray(joint_errors, dtype=np.float32)

    mean_error = float(joint_errors.mean())
    max_error = float(joint_errors.max())

    base_score_by_type = {
        "clean": 98.0,
        "noise_small": 92.0,
        "noise_large": 78.0,
        "timing_delay": 76.0,
        "left_arm_low": 74.0,
        "right_arm_low": 74.0,
        "leg_shift": 72.0,
        "upper_body_shift": 75.0,
    }

    base = base_score_by_type.get(error_type, 80.0)

    # rule-based pseudo score
    # 실제 사람이 매긴 점수가 아니라, 기준 안무와 fake user의 차이로 만든 임시 라벨임
    score = base - mean_error * 120.0 - max_error * 25.0
    score = float(np.clip(score, 0.0, 100.0))

    return score


def generate_synthetic_dataset_from_windows(idol_windows, variants_per_window=8, seed=42):
    idol_windows = np.asarray(idol_windows, dtype=np.float32)

    if idol_windows.ndim != 4:
        raise ValueError(
            f"idol_windows must have shape (W, 30, 33, 3), got {idol_windows.shape}"
        )

    if idol_windows.shape[1:] != (30, 33, 3):
        raise ValueError(
            f"idol_windows shape after W must be (30, 33, 3), got {idol_windows.shape[1:]}"
        )

    rng = np.random.default_rng(seed)

    error_types = [
        "clean",
        "noise_small",
        "noise_large",
        "timing_delay",
        "left_arm_low",
        "right_arm_low",
        "leg_shift",
        "upper_body_shift",
    ]

    idol_seqs = []
    user_seqs = []
    scores = []
    joint_errors_all = []
    highlight_joints_all = []
    metadata = []

    sample_id = 0

    for window_index, idol_seq in enumerate(idol_windows):
        for variant_idx in range(int(variants_per_window)):
            error_type = error_types[variant_idx % len(error_types)]

            user_seq = perturb_sequence(idol_seq, error_type=error_type, rng=rng)
            joint_errors = compute_joint_errors(idol_seq, user_seq)
            highlight_joints = get_topk_highlight_joints(joint_errors, top_k=3)
            score = compute_score_from_error(joint_errors, error_type)

            idol_seqs.append(idol_seq)
            user_seqs.append(user_seq)
            scores.append(score)
            joint_errors_all.append(joint_errors)
            highlight_joints_all.append(highlight_joints)

            metadata.append({
                "sample_id": int(sample_id),
                "window_index": int(window_index),
                "error_type": error_type,
                "score": float(score),
                "highlight_joints": [int(x) for x in highlight_joints],
            })

            sample_id += 1

    return {
        "idol_seqs": np.stack(idol_seqs, axis=0).astype(np.float32),
        "user_seqs": np.stack(user_seqs, axis=0).astype(np.float32),
        "scores": np.asarray(scores, dtype=np.float32),
        "joint_errors": np.stack(joint_errors_all, axis=0).astype(np.float32),
        "highlight_joints": np.asarray(highlight_joints_all, dtype=np.int64),
        "metadata": metadata,
    }


def save_synthetic_dataset(dataset, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "dca_v2_love_shot_synthetic.npz"
    metadata_path = output_dir / "synthetic_metadata.json"

    np.savez_compressed(
        npz_path,
        idol_seqs=dataset["idol_seqs"],
        user_seqs=dataset["user_seqs"],
        scores=dataset["scores"],
        joint_errors=dataset["joint_errors"],
        highlight_joints=dataset["highlight_joints"],
    )

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(dataset["metadata"], f, ensure_ascii=False, indent=2)

    return npz_path, metadata_path
