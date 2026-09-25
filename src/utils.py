import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def score_to_color(score_100, green_threshold=80.0, orange_threshold=60.0):
    score_100 = float(score_100)

    if score_100 >= green_threshold:
        return "green"

    if score_100 >= orange_threshold:
        return "orange"

    return "red"


def summarize_array(name, arr):
    arr = np.asarray(arr)

    print(f"[{name}]")
    print(f"  shape     : {arr.shape}")
    print(f"  dtype     : {arr.dtype}")

    if arr.size == 0:
        print("  empty array")
        return

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())

    print(f"  min       : {np.nanmin(arr):.6f}")
    print(f"  max       : {np.nanmax(arr):.6f}")
    print(f"  mean      : {np.nanmean(arr):.6f}")
    print(f"  std       : {np.nanstd(arr):.6f}")
    print(f"  NaN count : {nan_count}")
    print(f"  Inf count : {inf_count}")


def make_dummy_pose_sequence(
    frame_length=30,
    num_joints=33,
    coord_dim=3,
    noise_scale=0.01,
    seed=42,
):
    rng = np.random.default_rng(seed)

    base_pose = rng.normal(
        loc=0.0,
        scale=0.2,
        size=(num_joints, coord_dim),
    ).astype(np.float32)

    if num_joints > 24 and coord_dim >= 3:
        base_pose[11] = np.array([-0.25, 0.45, 0.00], dtype=np.float32)
        base_pose[12] = np.array([0.25, 0.45, 0.00], dtype=np.float32)
        base_pose[23] = np.array([-0.18, 0.00, 0.00], dtype=np.float32)
        base_pose[24] = np.array([0.18, 0.00, 0.00], dtype=np.float32)

    seq = []

    for t in range(frame_length):
        phase = 2.0 * np.pi * t / max(frame_length - 1, 1)

        global_motion = np.array(
            [
                0.03 * np.sin(phase),
                0.02 * np.cos(phase),
                0.01 * np.sin(phase * 0.5),
            ],
            dtype=np.float32,
        )

        joint_offsets = rng.normal(
            loc=0.0,
            scale=noise_scale,
            size=(num_joints, coord_dim),
        ).astype(np.float32)

        frame = base_pose + global_motion + joint_offsets
        seq.append(frame)

    return np.stack(seq, axis=0).astype(np.float32)


def make_dummy_pose_batch(
    batch_size=4,
    frame_length=30,
    num_joints=33,
    coord_dim=3,
    seed=42,
):
    batch = []

    for i in range(batch_size):
        seq = make_dummy_pose_sequence(
            frame_length=frame_length,
            num_joints=num_joints,
            coord_dim=coord_dim,
            seed=seed + i,
        )

        batch.append(seq)

    return np.stack(batch, axis=0).astype(np.float32)
