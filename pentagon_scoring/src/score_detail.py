"""Rule-based endpoint-position and limb-angle detail scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from pentagon_common import (
    clip_score,
    compute_joint_angles,
    compute_weighted_joint_distances,
    validate_pose_sequence,
)
from pentagon_config import PentagonConfig
from score_accuracy import MEDIAPIPE_POSE_JOINT_NAMES


DETAIL_POSITION_JOINT_INDICES: tuple[int, ...] = (
    15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32,
)
DETAIL_POSITION_WEIGHTS: tuple[float, ...] = (
    1.20, 1.20, 0.70, 0.70, 0.80, 0.80, 0.70, 0.70,
    1.20, 1.20, 0.90, 0.90, 1.00, 1.00,
)
DETAIL_ANGLE_SPECS: tuple[tuple[str, int, int, int], ...] = (
    ("left_elbow", 11, 13, 15),
    ("right_elbow", 12, 14, 16),
    ("left_knee", 23, 25, 27),
    ("right_knee", 24, 26, 28),
)
DETAIL_ANGLE_WEIGHTS: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)

# MVP initial calibration values; real videos and human ratings must tune them.
DETAIL_POSITION_SCALE: float = 350.0
DETAIL_ANGLE_SCALE: float = 1.5
DETAIL_POSITION_COMPONENT_WEIGHT: float = 0.60
DETAIL_ANGLE_COMPONENT_WEIGHT: float = 0.40


@dataclass(frozen=True)
class DetailDiagnostics:
    """Explain endpoint-position and joint-angle detail deductions."""

    position_score: float
    angle_score: float | None
    weighted_position_error: float
    weighted_angle_error_deg: float | None
    position_scale: float
    angle_scale: float
    position_component_weight: float
    angle_component_weight: float
    frame_count: int
    position_joint_count: int
    angle_count: int
    position_weight_sum: float
    effective_angle_weight_sum: float
    use_z: bool
    z_weight: float
    top_k: int
    per_position_joint_errors: list[float]
    per_position_joint_weights: list[float]
    per_position_joint_contributions: list[float]
    per_frame_position_errors: list[float]
    position_joint_indices: list[int]
    position_joint_names: list[str]
    worst_position_joint_indices: list[int]
    worst_position_joints: list[dict[str, Any]]
    angle_names: list[str]
    per_angle_errors_deg: list[float | None]
    per_angle_weights: list[float]
    per_angle_valid_frame_counts: list[int]
    per_angle_contributions: list[float]
    worst_angles: list[dict[str, Any]]
    reliable: bool
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostics containing only JSON-compatible Python values."""
        return asdict(self)


@dataclass(frozen=True)
class DetailScoreResult:
    """Detail score and its endpoint/angle diagnostics."""

    score: float
    diagnostics: DetailDiagnostics

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible score dictionary."""
        return {
            "score": float(self.score),
            "diagnostics": self.diagnostics.to_dict(),
        }


def _validate_top_k(top_k: int) -> int:
    maximum = len(DETAIL_POSITION_JOINT_INDICES)
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)):
        raise ValueError(f"top_k must be an integer, not bool; got {top_k!r}")
    value = int(top_k)
    if value < 1 or value > maximum:
        raise ValueError(f"top_k must be in [1, {maximum}]; got {value}")
    return value


def compute_detail_score(
    aligned_user_seq: Any,
    aligned_idol_seq: Any,
    config: PentagonConfig | None = None,
    top_k: int = 5,
) -> DetailScoreResult:
    """Score timing-aligned normalized poses of shape ``(T, 33, 3)``.

    Fourteen endpoint positions use the common xyz-weighted distance contract.
    Four A-B-C angles (B is the vertex) use only frames where both poses have
    valid vectors. The inputs are not normalized or timing-aligned again.
    ``T`` may differ from the original window but must meet
    ``minimum_aligned_frames``.
    """
    active_config = PentagonConfig() if config is None else config
    active_config.validate()
    selected_top_k = _validate_top_k(top_k)
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

    all_distances = compute_weighted_joint_distances(user, idol, aligned_config)
    expected_shape = (frame_count, active_config.num_joints)
    if all_distances.shape != expected_shape:
        raise ValueError(
            f"distance array must have shape {expected_shape}; "
            f"got {all_distances.shape}"
        )
    position_indices = np.asarray(DETAIL_POSITION_JOINT_INDICES, dtype=np.intp)
    position_distances = all_distances[:, position_indices].astype(
        np.float64, copy=False
    )
    position_weights = np.asarray(DETAIL_POSITION_WEIGHTS, dtype=np.float64)
    if position_weights.shape != position_indices.shape:
        raise ValueError(
            "DETAIL_POSITION_WEIGHTS length must match position indices; "
            f"got {position_weights.size} and {position_indices.size}"
        )
    position_weight_sum = float(np.sum(position_weights, dtype=np.float64))
    if position_weight_sum <= 0.0:
        raise ValueError(
            f"detail position weight sum must be > 0; got {position_weight_sum}"
        )
    per_position_errors = np.mean(position_distances, axis=0, dtype=np.float64)
    position_contributions = (
        position_weights * per_position_errors / position_weight_sum
    )
    weighted_position_error = float(
        np.sum(position_contributions, dtype=np.float64)
    )
    per_frame_position_errors = (
        np.sum(
            position_distances * position_weights[None, :],
            axis=1,
            dtype=np.float64,
        )
        / position_weight_sum
    )
    position_score = clip_score(
        100.0 - weighted_position_error * DETAIL_POSITION_SCALE
    )

    triplets = tuple(spec[1:] for spec in DETAIL_ANGLE_SPECS)
    user_angle_result = compute_joint_angles(user, triplets, aligned_config)
    idol_angle_result = compute_joint_angles(idol, triplets, aligned_config)
    pair_valid = user_angle_result.valid_mask & idol_angle_result.valid_mask
    angle_count = len(DETAIL_ANGLE_SPECS)
    angle_weights = np.asarray(DETAIL_ANGLE_WEIGHTS, dtype=np.float64)
    valid_counts = np.sum(pair_valid, axis=0, dtype=np.int64)
    per_angle_errors: list[float | None] = []
    for angle_index in range(angle_count):
        valid = pair_valid[:, angle_index]
        if valid_counts[angle_index] == 0:
            per_angle_errors.append(None)
        else:
            differences = np.abs(
                user_angle_result.angles[valid, angle_index].astype(np.float64)
                - idol_angle_result.angles[valid, angle_index].astype(np.float64)
            )
            per_angle_errors.append(float(np.mean(differences, dtype=np.float64)))

    effective_weights = np.where(valid_counts > 0, angle_weights, 0.0)
    effective_weight_sum = float(np.sum(effective_weights, dtype=np.float64))
    if effective_weight_sum == 0.0:
        weighted_angle_error: float | None = None
        angle_contributions = np.zeros(angle_count, dtype=np.float64)
        angle_score: float | None = None
        detail_score = position_score
        reliable = False
        fallback_reason: str | None = "all_angles_invalid"
    else:
        numeric_angle_errors = np.asarray(
            [0.0 if value is None else value for value in per_angle_errors],
            dtype=np.float64,
        )
        angle_contributions = (
            effective_weights * numeric_angle_errors / effective_weight_sum
        )
        weighted_angle_error = float(
            np.sum(angle_contributions, dtype=np.float64)
        )
        angle_score = clip_score(
            100.0 - weighted_angle_error * DETAIL_ANGLE_SCALE
        )
        detail_score = clip_score(
            position_score * DETAIL_POSITION_COMPONENT_WEIGHT
            + angle_score * DETAIL_ANGLE_COMPONENT_WEIGHT
        )
        reliable = True
        fallback_reason = None

    ranked_positions = sorted(
        range(len(DETAIL_POSITION_JOINT_INDICES)),
        key=lambda i: (
            -float(position_contributions[i]),
            DETAIL_POSITION_JOINT_INDICES[i],
        ),
    )
    worst_position_slots = ranked_positions[:selected_top_k]
    worst_position_indices = [
        DETAIL_POSITION_JOINT_INDICES[slot] for slot in worst_position_slots
    ]
    worst_position_joints = [
        {
            "index": int(DETAIL_POSITION_JOINT_INDICES[slot]),
            "name": MEDIAPIPE_POSE_JOINT_NAMES[
                DETAIL_POSITION_JOINT_INDICES[slot]
            ],
            "mean_error": float(per_position_errors[slot]),
            "weight": float(position_weights[slot]),
            "weighted_contribution": float(position_contributions[slot]),
        }
        for slot in worst_position_slots
    ]

    valid_angle_slots = [
        index for index in range(angle_count) if valid_counts[index] > 0
    ]
    ranked_angles = sorted(
        valid_angle_slots,
        key=lambda i: (-float(angle_contributions[i]), i),
    )[: min(selected_top_k, angle_count)]
    worst_angles = [
        {
            "name": DETAIL_ANGLE_SPECS[index][0],
            "points": [int(point) for point in DETAIL_ANGLE_SPECS[index][1:]],
            "mean_error_deg": float(per_angle_errors[index]),  # type: ignore[arg-type]
            "weight": float(angle_weights[index]),
            "weighted_contribution_deg": float(angle_contributions[index]),
            "valid_frame_count": int(valid_counts[index]),
        }
        for index in ranked_angles
    ]

    diagnostics = DetailDiagnostics(
        position_score=float(position_score),
        angle_score=None if angle_score is None else float(angle_score),
        weighted_position_error=weighted_position_error,
        weighted_angle_error_deg=weighted_angle_error,
        position_scale=float(DETAIL_POSITION_SCALE),
        angle_scale=float(DETAIL_ANGLE_SCALE),
        position_component_weight=float(DETAIL_POSITION_COMPONENT_WEIGHT),
        angle_component_weight=float(DETAIL_ANGLE_COMPONENT_WEIGHT),
        frame_count=frame_count,
        position_joint_count=len(DETAIL_POSITION_JOINT_INDICES),
        angle_count=angle_count,
        position_weight_sum=position_weight_sum,
        effective_angle_weight_sum=effective_weight_sum,
        use_z=bool(active_config.use_z),
        z_weight=float(active_config.z_weight),
        top_k=selected_top_k,
        per_position_joint_errors=[
            float(value) for value in per_position_errors
        ],
        per_position_joint_weights=[
            float(value) for value in position_weights
        ],
        per_position_joint_contributions=[
            float(value) for value in position_contributions
        ],
        per_frame_position_errors=[
            float(value) for value in per_frame_position_errors
        ],
        position_joint_indices=[
            int(value) for value in DETAIL_POSITION_JOINT_INDICES
        ],
        position_joint_names=[
            MEDIAPIPE_POSE_JOINT_NAMES[index]
            for index in DETAIL_POSITION_JOINT_INDICES
        ],
        worst_position_joint_indices=[
            int(value) for value in worst_position_indices
        ],
        worst_position_joints=worst_position_joints,
        angle_names=[spec[0] for spec in DETAIL_ANGLE_SPECS],
        per_angle_errors_deg=per_angle_errors,
        per_angle_weights=[float(value) for value in angle_weights],
        per_angle_valid_frame_counts=[
            int(value) for value in valid_counts
        ],
        per_angle_contributions=[
            float(value) for value in angle_contributions
        ],
        worst_angles=worst_angles,
        reliable=bool(reliable),
        fallback_reason=fallback_reason,
    )
    return DetailScoreResult(score=float(detail_score), diagnostics=diagnostics)
