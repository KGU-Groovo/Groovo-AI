"""Rule-based pose-motion rhythm scoring after global timing alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from .pentagon_common import (
    clip_score,
    compute_motion_signal,
    normalize_signal,
    validate_pose_sequence,
)
from .pentagon_config import PentagonConfig


@dataclass(frozen=True)
class RhythmDiagnostics:
    motion_correlation: float | None
    normalized_motion_mae: float
    correlation_score: float | None
    mae_score: float
    correlation_weight: float
    mae_weight: float
    rhythm_mae_scale: float
    frame_count: int
    aligned_motion_length: int
    user_motion_mean: float
    idol_motion_mean: float
    user_motion_std: float
    idol_motion_std: float
    user_constant_signal: bool
    idol_constant_signal: bool
    user_motion_signal: list[float]
    idol_motion_signal: list[float]
    normalized_user_motion_signal: list[float]
    normalized_idol_motion_signal: list[float]
    per_motion_abs_errors: list[float]
    reliable: bool
    fallback_reason: str | None
    rhythm_definition: str
    analyzes_audio: bool
    analyzes_bpm: bool

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostics containing only JSON-compatible Python values."""
        return asdict(self)


@dataclass(frozen=True)
class RhythmScoreResult:
    score: float
    diagnostics: RhythmDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "diagnostics": self.diagnostics.to_dict(),
        }


def _safe_pearson_correlation(
    first: np.ndarray,
    second: np.ndarray,
    epsilon: float,
) -> float | None:
    """Return explicitly denominator-checked Pearson correlation."""
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if (
        first_array.ndim != 1
        or second_array.ndim != 1
        or first_array.shape != second_array.shape
    ):
        raise ValueError("correlation inputs must be same-length 1-D arrays")
    if not np.isfinite(first_array).all() or not np.isfinite(second_array).all():
        raise ValueError("correlation inputs must be finite")
    first_centered = first_array - np.mean(first_array, dtype=np.float64)
    second_centered = second_array - np.mean(second_array, dtype=np.float64)
    denominator = float(
        np.sqrt(
            np.sum(first_centered**2, dtype=np.float64)
            * np.sum(second_centered**2, dtype=np.float64)
        )
    )
    if denominator <= epsilon or not np.isfinite(denominator):
        return None
    correlation = float(
        np.sum(first_centered * second_centered, dtype=np.float64) / denominator
    )
    if not np.isfinite(correlation):
        return None
    return float(np.clip(correlation, -1.0, 1.0))


def compute_rhythm_score(
    aligned_user_seq: Any,
    aligned_idol_seq: Any,
    config: PentagonConfig | None = None,
) -> RhythmScoreResult:
    """Score motion-pattern similarity in aligned, normalized pose sequences."""
    active_config = PentagonConfig() if config is None else config
    active_config.validate()
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

    user_motion = compute_motion_signal(user, aligned_config)
    idol_motion = compute_motion_signal(idol, aligned_config)
    if user_motion.shape != idol_motion.shape:
        raise ValueError(
            "user and idol motion signals must have identical shapes; "
            f"got {user_motion.shape} and {idol_motion.shape}"
        )
    if user_motion.ndim != 1 or user_motion.size < 1:
        raise ValueError(
            "motion signal must be a nonempty 1-D array; "
            f"got shape={user_motion.shape}"
        )

    user_stats = normalize_signal(user_motion, aligned_config)
    idol_stats = normalize_signal(idol_motion, aligned_config)
    normalized_user = user_stats.signal
    normalized_idol = idol_stats.signal
    if normalized_user.shape != user_motion.shape:
        raise ValueError(
            "normalized user motion shape must match source motion shape; "
            f"got {normalized_user.shape} and {user_motion.shape}"
        )
    if normalized_idol.shape != idol_motion.shape:
        raise ValueError(
            "normalized idol motion shape must match source motion shape; "
            f"got {normalized_idol.shape} and {idol_motion.shape}"
        )

    per_errors = np.abs(
        normalized_user.astype(np.float64)
        - normalized_idol.astype(np.float64)
    )
    normalized_mae = float(np.mean(per_errors, dtype=np.float64))
    mae_score = float(
        clip_score(
            100.0 - normalized_mae * float(active_config.rhythm_mae_scale)
        )
    )

    correlation: float | None = None
    correlation_score: float | None = None
    if user_stats.constant_signal and idol_stats.constant_signal:
        per_errors = np.zeros(user_motion.shape, dtype=np.float64)
        normalized_mae = 0.0
        mae_score = 100.0
        score = 100.0
        reliable = False
        fallback_reason: str | None = "both_constant"
    elif user_stats.constant_signal != idol_stats.constant_signal:
        score = 0.0
        reliable = False
        fallback_reason = "one_constant"
    else:
        correlation = _safe_pearson_correlation(
            normalized_user, normalized_idol, float(active_config.epsilon)
        )
        if correlation is None:
            score = mae_score
            reliable = False
            fallback_reason = "correlation_unavailable"
        else:
            correlation_score = float(
                clip_score((correlation + 1.0) / 2.0 * 100.0)
            )
            score = float(
                clip_score(
                    float(active_config.rhythm_correlation_weight)
                    * correlation_score
                    + float(active_config.rhythm_mae_weight) * mae_score
                )
            )
            reliable = True
            fallback_reason = None

    diagnostics = RhythmDiagnostics(
        motion_correlation=None if correlation is None else float(correlation),
        normalized_motion_mae=float(normalized_mae),
        correlation_score=(
            None if correlation_score is None else float(correlation_score)
        ),
        mae_score=float(mae_score),
        correlation_weight=float(active_config.rhythm_correlation_weight),
        mae_weight=float(active_config.rhythm_mae_weight),
        rhythm_mae_scale=float(active_config.rhythm_mae_scale),
        frame_count=frame_count,
        aligned_motion_length=int(user_motion.size),
        user_motion_mean=float(user_stats.mean),
        idol_motion_mean=float(idol_stats.mean),
        user_motion_std=float(user_stats.std),
        idol_motion_std=float(idol_stats.std),
        user_constant_signal=bool(user_stats.constant_signal),
        idol_constant_signal=bool(idol_stats.constant_signal),
        user_motion_signal=[float(value) for value in user_motion],
        idol_motion_signal=[float(value) for value in idol_motion],
        normalized_user_motion_signal=[
            float(value) for value in normalized_user
        ],
        normalized_idol_motion_signal=[
            float(value) for value in normalized_idol
        ],
        per_motion_abs_errors=[float(value) for value in per_errors],
        reliable=bool(reliable),
        fallback_reason=fallback_reason,
        rhythm_definition="pose_motion_similarity_after_timing_alignment",
        analyzes_audio=False,
        analyzes_bpm=False,
    )
    return RhythmScoreResult(score=float(clip_score(score)), diagnostics=diagnostics)
