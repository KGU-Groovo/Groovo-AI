from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .pentagon_config import MEDIAPIPE_POSE, PentagonConfig


@dataclass(frozen=True)
class NormalizationDiagnostics:
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    scale_factor: float
    valid_shoulder_width_frames: int
    total_frames: int
    mean_hip_center_before: tuple[float, float, float]
    median_shoulder_width_after: float
    coordinate_contract: str = "xyz"
    scale_plane: str = "xy"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairPreparationDiagnostics:
    already_normalized: bool
    user: NormalizationDiagnostics | None
    idol: NormalizationDiagnostics | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AngleResult:
    angles: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class SignalNormalizationResult:
    signal: np.ndarray
    constant_signal: bool
    mean: float
    std: float


def to_float32_array(sequence: Any, name: str = "sequence") -> np.ndarray:
    """숫자형 입력을 C-contiguous float32 배열로 변환한다."""
    try:
        array = np.asarray(sequence)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} cannot be converted to NumPy array"
        ) from exc

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"{name} must be numeric; dtype={array.dtype}, shape={array.shape}"
        )

    try:
        return np.ascontiguousarray(array, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} cannot be converted to float32; shape={array.shape}"
        ) from exc


def validate_pose_sequence(
    sequence: Any,
    config: PentagonConfig,
    name: str = "sequence",
) -> np.ndarray:
    """입력을 (T, 33, 3)의 유한한 xyz pose sequence로 검증한다."""
    array = to_float32_array(sequence, name)

    expected_shape = f"(T, {config.num_joints}, {config.coord_dim})"
    if array.ndim != 3:
        raise ValueError(
            f"{name} must have shape {expected_shape}; got shape={array.shape}"
        )
    if array.shape[1] != config.num_joints:
        raise ValueError(
            f"{name} must have {config.num_joints} joints; "
            f"got shape={array.shape}"
        )
    if array.shape[2] != config.coord_dim:
        raise ValueError(
            f"{name} must have {config.coord_dim} coordinates [x, y, z]; "
            f"got shape={array.shape}"
        )
    if array.shape[0] < 2:
        raise ValueError(
            f"{name} must contain at least 2 frames; got shape={array.shape}"
        )
    if (
        config.expected_frames is not None
        and array.shape[0] != config.expected_frames
    ):
        raise ValueError(
            f"{name} must contain {config.expected_frames} frames; "
            f"got {array.shape[0]} frames, shape={array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} must not contain NaN or Inf; shape={array.shape}"
        )

    return np.ascontiguousarray(array, dtype=np.float32)


def _hip_center_validated(sequence: np.ndarray) -> np.ndarray:
    left = sequence[:, MEDIAPIPE_POSE.left_hip : MEDIAPIPE_POSE.left_hip + 1]
    right = sequence[:, MEDIAPIPE_POSE.right_hip : MEDIAPIPE_POSE.right_hip + 1]
    return np.ascontiguousarray(0.5 * (left + right), dtype=np.float32)


def compute_hip_center(sequence: Any) -> np.ndarray:
    """프레임별 왼쪽/오른쪽 골반 중점을 (T, 1, 3)으로 반환한다."""
    config = PentagonConfig(expected_frames=None)
    array = validate_pose_sequence(sequence, config)
    return _hip_center_validated(array)


def _shoulder_widths_validated(sequence: np.ndarray) -> np.ndarray:
    delta_xy = (
        sequence[:, MEDIAPIPE_POSE.right_shoulder, :2]
        - sequence[:, MEDIAPIPE_POSE.left_shoulder, :2]
    )
    return np.ascontiguousarray(
        np.linalg.norm(delta_xy, axis=1),
        dtype=np.float32,
    )


def compute_shoulder_widths(
    sequence: Any,
    config: PentagonConfig,
) -> np.ndarray:
    """프레임별 양쪽 어깨의 xy 평면 거리를 (T,)로 반환한다."""
    array = validate_pose_sequence(sequence, config)
    return _shoulder_widths_validated(array)


def normalize_pose_sequence(
    sequence: Any,
    config: PentagonConfig,
    name: str = "sequence",
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, NormalizationDiagnostics]:
    """
    프레임별 골반 중심을 뺀 후 전체 sequence의 어깨너비 중앙값으로 나눈다.

    입력 채널은 [x, y, z] 계약이다. 어깨너비는 xy만 사용하고,
    x/y/z 좌표 모두 동일한 scale로 나눈다.
    """
    array = validate_pose_sequence(sequence, config, name)

    hip_centers = _hip_center_validated(array)
    centered = array - hip_centers

    widths = _shoulder_widths_validated(array)
    valid_mask = np.isfinite(widths) & (
        widths >= config.min_shoulder_width
    )
    valid_widths = widths[valid_mask]

    if valid_widths.size == 0:
        raise ValueError(
            f"{name} has no valid xy shoulder widths >= "
            f"{config.min_shoulder_width}; shape={array.shape}"
        )

    scale_factor = float(np.median(valid_widths))
    if (
        not np.isfinite(scale_factor)
        or scale_factor < config.min_shoulder_width
    ):
        raise ValueError(
            f"{name} has invalid shoulder scale={scale_factor}"
        )

    normalized = np.ascontiguousarray(
        centered / scale_factor,
        dtype=np.float32,
    )

    if not return_diagnostics:
        return normalized

    normalized_widths = _shoulder_widths_validated(normalized)
    diagnostics = NormalizationDiagnostics(
        input_shape=tuple(array.shape),
        output_shape=tuple(normalized.shape),
        scale_factor=scale_factor,
        valid_shoulder_width_frames=int(valid_mask.sum()),
        total_frames=int(array.shape[0]),
        mean_hip_center_before=tuple(
            float(v) for v in hip_centers.mean(axis=(0, 1))
        ),
        median_shoulder_width_after=float(
            np.median(normalized_widths[valid_mask])
        ),
    )
    return normalized, diagnostics


def prepare_pose_pair(
    user_seq: Any,
    idol_seq: Any,
    config: PentagonConfig,
    already_normalized: bool = False,
    return_diagnostics: bool = False,
):
    """두 sequence를 검증하고 필요하면 각각 독립적으로 정규화한다."""
    user = validate_pose_sequence(user_seq, config, "user_seq")
    idol = validate_pose_sequence(idol_seq, config, "idol_seq")

    if user.shape != idol.shape:
        raise ValueError(
            "user_seq and idol_seq must have identical shapes; "
            f"user={user.shape}, idol={idol.shape}"
        )

    user_diag = None
    idol_diag = None

    if not already_normalized:
        user, user_diag = normalize_pose_sequence(
            user,
            config,
            "user_seq",
            return_diagnostics=True,
        )
        idol, idol_diag = normalize_pose_sequence(
            idol,
            config,
            "idol_seq",
            return_diagnostics=True,
        )

    if return_diagnostics:
        diagnostics = PairPreparationDiagnostics(
            already_normalized=already_normalized,
            user=user_diag,
            idol=idol_diag,
        )
        return user, idol, diagnostics

    return user, idol


def _validate_joint_indices(
    joint_indices: Sequence[int] | None,
    config: PentagonConfig,
) -> np.ndarray:
    values = (
        tuple(range(config.num_joints))
        if joint_indices is None
        else tuple(joint_indices)
    )

    if not values:
        raise ValueError("joint_indices must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("joint_indices must be unique")
    if any(
        not isinstance(i, (int, np.integer))
        or isinstance(i, bool)
        or i < 0
        or i >= config.num_joints
        for i in values
    ):
        raise ValueError("joint_indices must be integers in [0, 32]")

    return np.asarray(values, dtype=np.intp)


def compute_weighted_joint_distances(
    user_seq: Any,
    idol_seq: Any,
    config: PentagonConfig,
    joint_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """
    관절별 거리 sqrt(dx² + dy² + z_weight*dz²)를 계산한다.

    아직 점수 변환은 하지 않는다.
    """
    user = validate_pose_sequence(user_seq, config, "user_seq")
    idol = validate_pose_sequence(idol_seq, config, "idol_seq")

    if user.shape != idol.shape:
        raise ValueError(
            f"shape mismatch: user={user.shape}, idol={idol.shape}"
        )

    indices = _validate_joint_indices(joint_indices, config)
    delta = user[:, indices] - idol[:, indices]

    squared = delta[..., 0] ** 2 + delta[..., 1] ** 2
    if config.use_z:
        squared += config.z_weight * delta[..., 2] ** 2

    return np.ascontiguousarray(
        np.sqrt(squared),
        dtype=np.float32,
    )


def compute_joint_angles(
    sequence: Any,
    angle_triplets: Sequence[tuple[int, int, int]],
    config: PentagonConfig,
) -> AngleResult:
    """
    A-B-C에서 B를 꼭짓점으로 하는 3차원 각도를 degree로 계산한다.

    길이가 거의 0인 벡터는 angles=NaN, valid_mask=False로 반환한다.
    """
    array = validate_pose_sequence(sequence, config)
    triplets = tuple(tuple(t) for t in angle_triplets)

    if not triplets:
        raise ValueError("angle_triplets must not be empty")
    if any(len(t) != 3 for t in triplets):
        raise ValueError("each angle triplet needs 3 indices")
    if any(
        not isinstance(i, (int, np.integer))
        or isinstance(i, bool)
        or i < 0
        or i >= config.num_joints
        for triplet in triplets
        for i in triplet
    ):
        raise ValueError("angle indices must be integers in [0, 32]")

    a_idx = np.asarray([t[0] for t in triplets], dtype=np.intp)
    b_idx = np.asarray([t[1] for t in triplets], dtype=np.intp)
    c_idx = np.asarray([t[2] for t in triplets], dtype=np.intp)

    u = array[:, a_idx] - array[:, b_idx]
    v = array[:, c_idx] - array[:, b_idx]

    norm_u = np.linalg.norm(u, axis=-1)
    norm_v = np.linalg.norm(v, axis=-1)
    denominator = norm_u * norm_v
    valid = denominator > config.epsilon

    cosine = np.zeros_like(denominator, dtype=np.float32)
    dot = np.sum(u * v, axis=-1)
    np.divide(dot, denominator, out=cosine, where=valid)
    np.clip(cosine, -1.0, 1.0, out=cosine)

    angles = np.full(cosine.shape, np.nan, dtype=np.float32)
    angles[valid] = np.degrees(np.arccos(cosine[valid]))

    return AngleResult(
        angles=np.ascontiguousarray(angles, dtype=np.float32),
        valid_mask=np.ascontiguousarray(valid),
    )


def compute_motion_signal(
    sequence: Any,
    config: PentagonConfig,
    joint_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """프레임 간 관절 이동거리의 가중 평균 신호를 (T-1,)로 반환한다."""
    array = validate_pose_sequence(sequence, config)

    weights = np.asarray(
        config.accuracy_joint_weights
        if joint_weights is None
        else joint_weights,
        dtype=np.float64,
    )

    if weights.shape != (config.num_joints,):
        raise ValueError(
            f"joint_weights shape must be ({config.num_joints},)"
        )
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("joint_weights must be finite and >= 0")

    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("joint_weights sum must be > 0")

    delta = np.diff(array, axis=0)
    squared = delta[..., 0] ** 2 + delta[..., 1] ** 2
    if config.use_z:
        squared += config.z_weight * delta[..., 2] ** 2

    distances = np.sqrt(squared)
    signal = np.sum(
        distances * weights[None, :],
        axis=1,
    ) / total_weight

    return np.ascontiguousarray(signal, dtype=np.float32)


def normalize_signal(
    signal: Any,
    config: PentagonConfig,
) -> SignalNormalizationResult:
    """1차원 신호를 표준화하고 정지 신호 여부를 반환한다."""
    array = to_float32_array(signal, "signal")

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"signal must be nonempty 1-D; shape={array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("signal must not contain NaN or Inf")

    mean = float(np.mean(array, dtype=np.float64))
    std = float(np.std(array, dtype=np.float64))

    if std <= config.epsilon:
        return SignalNormalizationResult(
            signal=np.zeros_like(array, dtype=np.float32),
            constant_signal=True,
            mean=mean,
            std=std,
        )

    normalized = (
        array.astype(np.float64) - mean
    ) / std

    return SignalNormalizationResult(
        signal=np.ascontiguousarray(normalized, dtype=np.float32),
        constant_signal=False,
        mean=mean,
        std=std,
    )


def clip_score(score: float) -> float:
    """유한한 scalar 점수를 0~100 범위로 제한한다."""
    array = np.asarray(score)

    if array.ndim != 0 or not np.issubdtype(array.dtype, np.number):
        raise ValueError("score must be numeric scalar")

    value = float(array)
    if not np.isfinite(value):
        raise ValueError("score must be finite")

    return float(np.clip(value, 0.0, 100.0))
