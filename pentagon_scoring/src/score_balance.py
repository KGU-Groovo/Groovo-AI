"""Rule-based frontal-2D balance scoring after timing alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from pentagon_common import clip_score, validate_pose_sequence
from pentagon_config import MEDIAPIPE_POSE, PentagonConfig


BALANCE_COMPONENT_NAMES: tuple[str, ...] = (
    "shoulder_tilt",
    "hip_tilt",
    "torso_lean",
    "support_center",
    "stance_width",
)
BALANCE_COMPONENT_WEIGHTS: tuple[float, ...] = (0.20, 0.20, 0.25, 0.20, 0.15)
BALANCE_COMPONENT_SCALES: tuple[float, ...] = (2.5, 2.5, 2.0, 250.0, 200.0)


@dataclass(frozen=True)
class BalanceDiagnostics:
    component_names: list[str]
    component_scores: list[float | None]
    component_mean_errors: list[float | None]
    component_weights: list[float]
    component_scales: list[float]
    effective_component_weights: list[float]
    component_valid_frame_counts: list[int]
    component_weighted_deductions: list[float]
    per_frame_shoulder_tilt_errors_deg: list[float | None]
    per_frame_hip_tilt_errors_deg: list[float | None]
    per_frame_torso_lean_errors_deg: list[float | None]
    per_frame_support_center_errors: list[float | None]
    per_frame_stance_width_errors: list[float | None]
    effective_component_weight_sum: float
    frame_count: int
    top_k: int
    worst_components: list[dict[str, Any]]
    reliable: bool
    fallback_reason: str | None
    camera_assumption: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BalanceScoreResult:
    score: float
    diagnostics: BalanceDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {"score": float(self.score), "diagnostics": self.diagnostics.to_dict()}


def _validate_top_k(top_k: int) -> int:
    maximum = len(BALANCE_COMPONENT_NAMES)
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)):
        raise ValueError(f"top_k must be an integer, not bool; got {top_k!r}")
    value = int(top_k)
    if value < 1 or value > maximum:
        raise ValueError(f"top_k must be in [1, {maximum}]; got {value}")
    return value


def _angles(vectors: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))


def _errors_with_mask(
    errors: np.ndarray, valid: np.ndarray
) -> tuple[list[float | None], float | None, int]:
    values = [
        float(errors[index]) if bool(valid[index]) else None
        for index in range(errors.shape[0])
    ]
    count = int(np.count_nonzero(valid))
    mean = None if count == 0 else float(np.mean(errors[valid], dtype=np.float64))
    return values, mean, count


def compute_balance_score(
    aligned_user_seq: Any,
    aligned_idol_seq: Any,
    config: PentagonConfig | None = None,
    top_k: int = 5,
) -> BalanceScoreResult:
    """Score aligned normalized poses using only frontal image-plane x and y."""
    active_config = PentagonConfig() if config is None else config
    active_config.validate()
    selected_top_k = _validate_top_k(top_k)
    aligned_config = replace(active_config, expected_frames=None)
    user = validate_pose_sequence(aligned_user_seq, aligned_config, "aligned_user_seq")
    idol = validate_pose_sequence(aligned_idol_seq, aligned_config, "aligned_idol_seq")
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

    ls, rs = MEDIAPIPE_POSE.left_shoulder, MEDIAPIPE_POSE.right_shoulder
    lh, rh = MEDIAPIPE_POSE.left_hip, MEDIAPIPE_POSE.right_hip
    la, ra = MEDIAPIPE_POSE.left_ankle, MEDIAPIPE_POSE.right_ankle
    eps = float(active_config.epsilon)

    def geometry(pose: np.ndarray) -> tuple[np.ndarray, ...]:
        shoulder = pose[:, rs, :2] - pose[:, ls, :2]
        hip = pose[:, rh, :2] - pose[:, lh, :2]
        shoulder_center = (pose[:, ls, :2] + pose[:, rs, :2]) * 0.5
        hip_center = (pose[:, lh, :2] + pose[:, rh, :2]) * 0.5
        torso = shoulder_center - hip_center
        ankle = pose[:, ra, :2] - pose[:, la, :2]
        ankle_center = (pose[:, la, :2] + pose[:, ra, :2]) * 0.5
        support = ankle_center[:, 0] - hip_center[:, 0]
        stance = np.abs(pose[:, ra, 0] - pose[:, la, 0])
        return shoulder, hip, torso, ankle, support, stance

    ug, ig = geometry(user), geometry(idol)
    valid_shoulder = (np.linalg.norm(ug[0], axis=1) > eps) & (
        np.linalg.norm(ig[0], axis=1) > eps
    )
    valid_hip = (np.linalg.norm(ug[1], axis=1) > eps) & (
        np.linalg.norm(ig[1], axis=1) > eps
    )
    valid_torso = (np.linalg.norm(ug[2], axis=1) > eps) & (
        np.linalg.norm(ig[2], axis=1) > eps
    )
    valid_ankle = (np.linalg.norm(ug[3], axis=1) > eps) & (
        np.linalg.norm(ig[3], axis=1) > eps
    )

    def axial(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        raw = np.abs(a - b) % 180.0
        return np.minimum(raw, 180.0 - raw)

    def circular(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.abs(((a - b + 180.0) % 360.0) - 180.0)

    raw_errors = (
        axial(_angles(ug[0]), _angles(ig[0])),
        axial(_angles(ug[1]), _angles(ig[1])),
        circular(_angles(ug[2]), _angles(ig[2])),
        np.abs(ug[4] - ig[4]),
        np.abs(ug[5] - ig[5]),
    )
    masks = (valid_shoulder, valid_hip, valid_torso, valid_ankle, valid_ankle)
    processed = [_errors_with_mask(error, mask) for error, mask in zip(raw_errors, masks)]
    per_frame = [item[0] for item in processed]
    mean_errors = [item[1] for item in processed]
    valid_counts = [item[2] for item in processed]
    scores = [
        None if error is None else float(clip_score(100.0 - error * scale))
        for error, scale in zip(mean_errors, BALANCE_COMPONENT_SCALES)
    ]
    effective_weights = [
        float(weight) if count > 0 else 0.0
        for weight, count in zip(BALANCE_COMPONENT_WEIGHTS, valid_counts)
    ]
    weight_sum = float(sum(effective_weights))
    if weight_sum == 0.0:
        score = 0.0
        reliable = False
        fallback_reason: str | None = "all_balance_components_invalid"
    else:
        score = float(
            sum(
                weight * (0.0 if component_score is None else component_score)
                for weight, component_score in zip(effective_weights, scores)
            )
            / weight_sum
        )
        score = float(clip_score(score))
        reliable = True
        fallback_reason = None
    deductions = [
        0.0
        if weight_sum == 0.0 or component_score is None
        else float(weight * (100.0 - component_score) / weight_sum)
        for weight, component_score in zip(effective_weights, scores)
    ]
    valid_slots = [index for index, count in enumerate(valid_counts) if count > 0]
    ranked = sorted(valid_slots, key=lambda index: (-deductions[index], index))
    worst = [
        {
            "name": BALANCE_COMPONENT_NAMES[index],
            "mean_error": float(mean_errors[index]),  # type: ignore[arg-type]
            "score": float(scores[index]),  # type: ignore[arg-type]
            "weight": float(BALANCE_COMPONENT_WEIGHTS[index]),
            "valid_frame_count": int(valid_counts[index]),
            "weighted_deduction": float(deductions[index]),
        }
        for index in ranked[: min(selected_top_k, len(BALANCE_COMPONENT_NAMES))]
    ]
    diagnostics = BalanceDiagnostics(
        component_names=list(BALANCE_COMPONENT_NAMES),
        component_scores=scores,
        component_mean_errors=mean_errors,
        component_weights=[float(value) for value in BALANCE_COMPONENT_WEIGHTS],
        component_scales=[float(value) for value in BALANCE_COMPONENT_SCALES],
        effective_component_weights=effective_weights,
        component_valid_frame_counts=valid_counts,
        component_weighted_deductions=deductions,
        per_frame_shoulder_tilt_errors_deg=per_frame[0],
        per_frame_hip_tilt_errors_deg=per_frame[1],
        per_frame_torso_lean_errors_deg=per_frame[2],
        per_frame_support_center_errors=per_frame[3],
        per_frame_stance_width_errors=per_frame[4],
        effective_component_weight_sum=weight_sum,
        frame_count=frame_count,
        top_k=selected_top_k,
        worst_components=worst,
        reliable=reliable,
        fallback_reason=fallback_reason,
        camera_assumption="frontal_2d",
    )
    return BalanceScoreResult(score=float(score), diagnostics=diagnostics)
