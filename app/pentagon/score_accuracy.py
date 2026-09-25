"""Rule-based pose accuracy scoring after global timing alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from .pentagon_common import (
    clip_score,
    compute_weighted_joint_distances,
    validate_pose_sequence,
)
from .pentagon_config import PentagonConfig


MEDIAPIPE_POSE_JOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


@dataclass(frozen=True)
class AccuracyDiagnostics:
    """Explain whole-body and per-joint accuracy deductions."""

    mean_joint_error: float
    weighted_mean_joint_error: float
    accuracy_scale: float
    use_z: bool
    z_weight: float
    frame_count: int
    joint_count: int
    joint_weight_sum: float
    top_k: int
    per_joint_errors: list[float]
    per_joint_weights: list[float]
    per_joint_weighted_contributions: list[float]
    per_frame_weighted_errors: list[float]
    worst_joint_indices: list[int]
    worst_joints: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary of Python scalar/container types."""
        return asdict(self)


@dataclass(frozen=True)
class AccuracyScoreResult:
    """Accuracy score and its JSON-compatible diagnostics."""

    score: float
    diagnostics: AccuracyDiagnostics

    def to_dict(self) -> dict[str, Any]:
        """Return ``{"score": float, "diagnostics": {...}}``."""
        return {
            "score": float(self.score),
            "diagnostics": self.diagnostics.to_dict(),
        }


def _validate_top_k(top_k: int, joint_count: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)):
        raise ValueError(f"top_k must be an integer, not bool; got {top_k!r}")
    value = int(top_k)
    if value < 1 or value > joint_count:
        raise ValueError(f"top_k must be in [1, {joint_count}]; got {value}")
    return value


def compute_accuracy_score(
    aligned_user_seq: Any,
    aligned_idol_seq: Any,
    config: PentagonConfig | None = None,
    top_k: int = 5,
) -> AccuracyScoreResult:
    """Score timing-aligned, normalized poses with shape ``(T, 33, 3)``.

    The inputs must already be independently normalized and globally aligned.
    This function neither normalizes poses nor estimates timing.  ``T`` may be
    shorter than the original window but must be at least
    ``minimum_aligned_frames``.

    Distances are ``sqrt(dx^2 + dy^2 + z_weight*dz^2)`` when z is enabled,
    otherwise ``sqrt(dx^2 + dy^2)``.  The configured 33 joint weights produce
    a weighted mean error, and the score is
    ``100 * exp(-weighted_mean_error * accuracy_scale)``.
    ``accuracy_scale`` and joint weights are MVP calibration values requiring
    adjustment with real user videos and human ratings.
    """
    active_config = PentagonConfig() if config is None else config
    active_config.validate()
    selected_top_k = _validate_top_k(top_k, active_config.num_joints)
    aligned_config = replace(active_config, expected_frames=None)

    user = validate_pose_sequence(
        aligned_user_seq, aligned_config, "aligned_user_seq"
    )
    idol = validate_pose_sequence(
        aligned_idol_seq, aligned_config, "aligned_idol_seq"
    )
    if user.shape != idol.shape:
        raise ValueError(
            "aligned_user_seq and aligned_idol_seq must have identical shapes; "
            f"got {user.shape} and {idol.shape}"
        )
    frame_count = int(user.shape[0])
    if frame_count < active_config.minimum_aligned_frames:
        raise ValueError(
            "aligned frame count must be at least "
            f"{active_config.minimum_aligned_frames}; got {frame_count}"
        )

    distances = compute_weighted_joint_distances(
        user, idol, aligned_config
    )
    expected_distance_shape = (frame_count, active_config.num_joints)
    if distances.shape != expected_distance_shape:
        raise ValueError(
            "distance array must have shape "
            f"{expected_distance_shape}; got {distances.shape}"
        )
    distances64 = distances.astype(np.float64, copy=False)
    per_joint_errors_array = np.mean(distances64, axis=0)
    mean_joint_error = float(np.mean(distances64))

    weights = np.asarray(active_config.accuracy_joint_weights, dtype=np.float64)
    if weights.shape != (active_config.num_joints,):
        raise ValueError(
            "accuracy_joint_weights must have shape "
            f"({active_config.num_joints},); got {weights.shape}"
        )
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("accuracy_joint_weights must be finite and nonnegative")
    weight_sum = float(np.sum(weights, dtype=np.float64))
    if weight_sum <= 0.0:
        raise ValueError(
            f"accuracy_joint_weights sum must be > 0; got {weight_sum}"
        )

    weighted_contributions = weights * per_joint_errors_array / weight_sum
    weighted_mean_error = float(np.sum(weighted_contributions, dtype=np.float64))
    per_frame_weighted_errors = (
        np.sum(distances64 * weights[None, :], axis=1, dtype=np.float64)
        / weight_sum
    )
    ranked_indices = sorted(
        range(active_config.num_joints),
        key=lambda index: (-float(weighted_contributions[index]), index),
    )
    worst_indices = ranked_indices[:selected_top_k]
    worst_joints = [
        {
            "index": int(index),
            "name": MEDIAPIPE_POSE_JOINT_NAMES[index],
            "mean_error": float(per_joint_errors_array[index]),
            "weight": float(weights[index]),
            "weighted_contribution": float(weighted_contributions[index]),
        }
        for index in worst_indices
    ]
    score = clip_score(
        100.0 * np.exp(-weighted_mean_error * float(active_config.accuracy_scale))
    )

    diagnostics = AccuracyDiagnostics(
        mean_joint_error=mean_joint_error,
        weighted_mean_joint_error=weighted_mean_error,
        accuracy_scale=float(active_config.accuracy_scale),
        use_z=bool(active_config.use_z),
        z_weight=float(active_config.z_weight),
        frame_count=frame_count,
        joint_count=int(active_config.num_joints),
        joint_weight_sum=weight_sum,
        top_k=selected_top_k,
        per_joint_errors=[float(value) for value in per_joint_errors_array],
        per_joint_weights=[float(value) for value in weights],
        per_joint_weighted_contributions=[
            float(value) for value in weighted_contributions
        ],
        per_frame_weighted_errors=[
            float(value) for value in per_frame_weighted_errors
        ],
        worst_joint_indices=[int(index) for index in worst_indices],
        worst_joints=worst_joints,
    )
    return AccuracyScoreResult(score=float(score), diagnostics=diagnostics)
