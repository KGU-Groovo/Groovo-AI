import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from config import (
        FRAME_LENGTH,
        NUM_JOINTS,
        COORD_DIM,
        LEFT_HIP_INDEX,
        RIGHT_HIP_INDEX,
        LEFT_SHOULDER_INDEX,
        RIGHT_SHOULDER_INDEX,
        EPS,
        INPUT_SHAPE,
    )
except ImportError:
    FRAME_LENGTH = 30
    NUM_JOINTS = 33
    COORD_DIM = 3
    LEFT_HIP_INDEX = 23
    RIGHT_HIP_INDEX = 24
    LEFT_SHOULDER_INDEX = 11
    RIGHT_SHOULDER_INDEX = 12
    EPS = 1e-8
    INPUT_SHAPE = (30, 33, 3)


def to_numpy_array(x, name="input"):
    if isinstance(x, np.ndarray):
        arr = x
    elif torch is not None and isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, list):
        arr = np.asarray(x)
    else:
        raise TypeError(
            f"{name} must be a list, numpy array, or torch tensor. Got {type(x)}"
        )

    return arr.astype(np.float32)


def check_finite(arr, name):
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values")

    if np.isinf(arr).any():
        raise ValueError(f"{name} contains Inf values")


def validate_pose_sequence(
    seq,
    name="seq",
    frame_length=FRAME_LENGTH,
    num_joints=NUM_JOINTS,
    coord_dim=COORD_DIM,
):
    arr = to_numpy_array(seq, name=name)
    expected_shape = (frame_length, num_joints, coord_dim)

    if arr.shape != expected_shape:
        raise ValueError(f"{name} shape must be {expected_shape}, got {arr.shape}")

    check_finite(arr, name)

    return arr.astype(np.float32)


def validate_pose_frame(
    frame,
    name="frame",
    num_joints=NUM_JOINTS,
    coord_dim=COORD_DIM,
):
    arr = to_numpy_array(frame, name=name)
    expected_shape = (num_joints, coord_dim)

    if arr.shape != expected_shape:
        raise ValueError(f"{name} shape must be {expected_shape}, got {arr.shape}")

    check_finite(arr, name)

    return arr.astype(np.float32)


def get_hip_center_from_frame(frame):
    frame = validate_pose_frame(frame, name="frame")

    left_hip = frame[LEFT_HIP_INDEX]
    right_hip = frame[RIGHT_HIP_INDEX]

    hip_center = (left_hip + right_hip) / 2.0

    return hip_center.astype(np.float32)


def get_shoulder_width_from_frame(frame):
    frame = validate_pose_frame(frame, name="frame")

    left_shoulder = frame[LEFT_SHOULDER_INDEX]
    right_shoulder = frame[RIGHT_SHOULDER_INDEX]

    width = float(np.linalg.norm(left_shoulder - right_shoulder))

    if width < EPS:
        width = float(EPS)

    return width


def normalize_pose_frame(frame):
    frame = validate_pose_frame(frame, name="frame")

    hip_center = get_hip_center_from_frame(frame)
    shoulder_width = get_shoulder_width_from_frame(frame)

    centered = frame - hip_center
    normalized = centered / shoulder_width

    return normalized.astype(np.float32)


def normalize_pose_sequence(seq):
    seq = validate_pose_sequence(seq, name="seq")
    hip_centers = (seq[:, LEFT_HIP_INDEX] + seq[:, RIGHT_HIP_INDEX]) / 2.0
    shoulder_widths = np.linalg.norm(
        seq[:, LEFT_SHOULDER_INDEX] - seq[:, RIGHT_SHOULDER_INDEX], axis=1
    )
    valid_widths = shoulder_widths[shoulder_widths > EPS]
    scale = float(np.median(valid_widths)) if len(valid_widths) else float(EPS)

    return ((seq - hip_centers[:, None, :]) / scale).astype(np.float32)


def normalize_pose_batch(batch):
    batch = to_numpy_array(batch, name="batch")

    if batch.ndim != 4:
        raise ValueError(
            f"batch must have 4 dimensions: (B, 30, 33, 3), got {batch.shape}"
        )

    if tuple(batch.shape[1:]) != tuple(INPUT_SHAPE):
        raise ValueError(
            f"batch shape after B must be {INPUT_SHAPE}, got {batch.shape[1:]}"
        )

    check_finite(batch, "batch")

    normalized_samples = []

    for sample_idx in range(batch.shape[0]):
        normalized_sample = normalize_pose_sequence(batch[sample_idx])
        normalized_samples.append(normalized_sample)

    return np.stack(normalized_samples, axis=0).astype(np.float32)


def compute_joint_errors(idol_seq, user_seq):
    idol_seq = validate_pose_sequence(idol_seq, name="idol_seq")
    user_seq = validate_pose_sequence(user_seq, name="user_seq")

    diff = idol_seq - user_seq

    frame_joint_dist = np.linalg.norm(diff, axis=2)

    joint_errors = frame_joint_dist.mean(axis=0)

    return joint_errors.astype(np.float32)


def get_topk_highlight_joints(joint_errors, top_k=3):
    errors = to_numpy_array(joint_errors, name="joint_errors")

    if errors.shape != (NUM_JOINTS,):
        raise ValueError(
            f"joint_errors shape must be ({NUM_JOINTS},), got {errors.shape}"
        )

    check_finite(errors, "joint_errors")

    top_k = int(top_k)
    top_k = max(1, min(top_k, NUM_JOINTS))

    indices = np.argsort(errors)[::-1][:top_k]

    return [int(idx) for idx in indices]


def get_highlight_coords(user_seq, highlight_joints, use_last_frame=True):
    user_seq = validate_pose_sequence(user_seq, name="user_seq")

    if use_last_frame:
        frame = user_seq[-1]
    else:
        frame = user_seq.mean(axis=0)

    coords = []

    for joint_idx in highlight_joints:
        joint_idx = int(joint_idx)

        if joint_idx < 0 or joint_idx >= NUM_JOINTS:
            raise ValueError(f"joint index out of range: {joint_idx}")

        coords.append(frame[joint_idx].astype(float).tolist())

    return coords
