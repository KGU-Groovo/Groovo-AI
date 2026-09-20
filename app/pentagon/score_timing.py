"""Rule-based global timing-lag scoring for normalized pose sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .pentagon_common import (
    clip_score,
    compute_motion_signal,
    normalize_signal,
    prepare_pose_pair,
    validate_pose_sequence,
)
from .pentagon_config import PentagonConfig


@dataclass(frozen=True)
class TimingDiagnostics:
    """Explain the selected global pose-motion lag and its reliability."""

    lag_frames: int
    lag_ms: float
    abs_lag_ms: float
    lag_direction: str
    correlation_at_best_lag: float | None
    max_lag_frames: int
    penalty_per_frame: float
    original_frame_count: int
    aligned_frame_count: int
    motion_overlap_count: int
    user_motion_std: float
    idol_motion_std: float
    user_constant_signal: bool
    idol_constant_signal: bool
    reliable: bool
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostics containing only JSON-compatible Python types."""
        return asdict(self)


@dataclass(frozen=True)
class TimingScoreResult:
    """Timing score, aligned ``(T-|lag|, 33, 3)`` poses, and diagnostics."""

    score: float
    aligned_user_seq: np.ndarray
    aligned_idol_seq: np.ndarray
    diagnostics: TimingDiagnostics

    def to_dict(self, include_aligned_sequences: bool = False) -> dict[str, Any]:
        """Return a JSON-compatible result, excluding large arrays by default."""
        result: dict[str, Any] = {
            "score": float(self.score),
            "diagnostics": self.diagnostics.to_dict(),
        }
        if include_aligned_sequences:
            result["aligned_user_seq"] = self.aligned_user_seq.tolist()
            result["aligned_idol_seq"] = self.aligned_idol_seq.tolist()
        return result


def _generate_lag_candidates(max_lag_frames: int) -> list[int]:
    if not isinstance(max_lag_frames, (int, np.integer)) or max_lag_frames < 0:
        raise ValueError(
            f"max_lag_frames must be a nonnegative integer; got {max_lag_frames}"
        )
    candidates = [0]
    for magnitude in range(1, int(max_lag_frames) + 1):
        candidates.extend((-magnitude, magnitude))
    return candidates


def _slice_signals_for_lag(
    user_motion: np.ndarray,
    idol_motion: np.ndarray,
    lag_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lag_frames > 0:
        return user_motion[lag_frames:], idol_motion[:-lag_frames]
    if lag_frames < 0:
        return user_motion[:lag_frames], idol_motion[-lag_frames:]
    return user_motion, idol_motion


def _safe_pearson_correlation(
    first: np.ndarray,
    second: np.ndarray,
    epsilon: float,
) -> float | None:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if (
        first_array.ndim != 1
        or second_array.ndim != 1
        or first_array.shape != second_array.shape
    ):
        raise ValueError("correlation inputs must be same-length 1-D arrays")
    if first_array.size < 2:
        return None
    if not np.isfinite(first_array).all() or not np.isfinite(second_array).all():
        return None
    first_centered = first_array - first_array.mean()
    second_centered = second_array - second_array.mean()
    first_std = float(np.std(first_array))
    second_std = float(np.std(second_array))
    if first_std <= epsilon or second_std <= epsilon:
        return None
    denominator = float(
        np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2))
    )
    if denominator <= epsilon or not np.isfinite(denominator):
        return None
    correlation = float(np.sum(first_centered * second_centered) / denominator)
    if not np.isfinite(correlation):
        return None
    return float(np.clip(correlation, -1.0, 1.0))


def _validate_motion_pair(
    user_motion: Any,
    idol_motion: Any,
    config: PentagonConfig,
) -> tuple[np.ndarray, np.ndarray]:
    user = np.asarray(user_motion)
    idol = np.asarray(idol_motion)
    if not np.issubdtype(user.dtype, np.number) or not np.issubdtype(
        idol.dtype, np.number
    ):
        raise ValueError("motion signals must have numeric dtypes")
    user = np.ascontiguousarray(user, dtype=np.float32)
    idol = np.ascontiguousarray(idol, dtype=np.float32)
    if user.ndim != 1 or idol.ndim != 1 or user.shape != idol.shape:
        raise ValueError(
            "user_motion and idol_motion must be same-length 1-D arrays; "
            f"got {user.shape} and {idol.shape}"
        )
    if user.size == 0 or not np.isfinite(user).all() or not np.isfinite(idol).all():
        raise ValueError("motion signals must be nonempty and finite")
    config.validate()
    return user, idol


def _select_best_lag(
    user_normalized: np.ndarray,
    idol_normalized: np.ndarray,
    config: PentagonConfig,
) -> tuple[int, float, int]:
    minimum_overlap = config.minimum_aligned_frames - 1
    best: tuple[int, float, int] | None = None
    tie_tolerance = max(config.epsilon, 1e-12)
    for lag in _generate_lag_candidates(config.max_lag_frames):
        user_overlap, idol_overlap = _slice_signals_for_lag(
            user_normalized, idol_normalized, lag
        )
        if user_overlap.size < minimum_overlap:
            continue
        correlation = _safe_pearson_correlation(
            user_overlap, idol_overlap, config.epsilon
        )
        if correlation is None:
            continue
        if best is None or correlation > best[1] + tie_tolerance:
            best = (int(lag), float(correlation), int(user_overlap.size))
    if best is None:
        raise ValueError(
            "no valid timing lag candidate: motion overlap must be at least "
            f"{minimum_overlap} and nonconstant for Pearson correlation"
        )
    return best


def estimate_timing_lag(
    user_motion: Any,
    idol_motion: Any,
    config: PentagonConfig,
) -> tuple[int, float, int]:
    """Estimate lag from same-length ``(T-1,)`` motion signals.

    Returns:
        ``(lag_frames, correlation_at_best_lag, motion_overlap_count)``.

    Raises:
        ValueError: If signals are invalid/constant or no candidate has the
            required ``minimum_aligned_frames - 1`` overlap.
    """
    user, idol = _validate_motion_pair(user_motion, idol_motion, config)
    user_stats = normalize_signal(user, config)
    idol_stats = normalize_signal(idol, config)
    if user_stats.constant_signal or idol_stats.constant_signal:
        raise ValueError("timing lag is unobservable for a constant motion signal")
    return _select_best_lag(user_stats.signal, idol_stats.signal, config)


def align_pose_sequences(
    user_seq: Any,
    idol_seq: Any,
    lag_frames: int,
    config: PentagonConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Align equal ``(T, 33, 3)`` poses using the signed timing lag contract."""
    user = validate_pose_sequence(user_seq, config, "user_seq")
    idol = validate_pose_sequence(idol_seq, config, "idol_seq")
    if user.shape != idol.shape:
        raise ValueError(
            "user_seq and idol_seq must have identical shapes; "
            f"got {user.shape} and {idol.shape}"
        )
    if not isinstance(lag_frames, (int, np.integer)):
        raise ValueError(f"lag_frames must be an integer; got {lag_frames}")
    lag = int(lag_frames)
    frame_count = int(user.shape[0])
    if abs(lag) >= frame_count:
        raise ValueError(
            f"abs(lag_frames) must be less than frame count {frame_count}; got {lag}"
        )
    if lag > 0:
        aligned_user, aligned_idol = user[lag:], idol[:-lag]
    elif lag < 0:
        aligned_user, aligned_idol = user[:lag], idol[-lag:]
    else:
        aligned_user, aligned_idol = user, idol
    if aligned_user.shape[0] < config.minimum_aligned_frames:
        raise ValueError(
            "aligned frame count must be at least "
            f"{config.minimum_aligned_frames}; got {aligned_user.shape[0]}"
        )
    if aligned_user.shape != aligned_idol.shape:
        raise ValueError("aligned pose sequence shapes must be identical")
    return (
        np.ascontiguousarray(aligned_user, dtype=np.float32),
        np.ascontiguousarray(aligned_idol, dtype=np.float32),
    )


def compute_timing_score(
    user_seq: Any,
    idol_seq: Any,
    config: PentagonConfig | None = None,
    already_normalized: bool = False,
) -> TimingScoreResult:
    """Score and align equal pose sequences with shape ``(T, 33, 3)``.

    Inputs are independently normalized exactly once unless
    ``already_normalized=True``, in which case they are validated only.
    """
    active_config = PentagonConfig() if config is None else config
    user, idol = prepare_pose_pair(
        user_seq,
        idol_seq,
        active_config,
        already_normalized=already_normalized,
    )
    user_motion = compute_motion_signal(user, active_config)
    idol_motion = compute_motion_signal(idol, active_config)
    user_stats = normalize_signal(user_motion, active_config)
    idol_stats = normalize_signal(idol_motion, active_config)

    fallback_reason: str | None = None
    correlation: float | None = None
    reliable = False
    if user_stats.constant_signal and idol_stats.constant_signal:
        lag, score = 0, 100.0
        direction = "unobservable"
        overlap_count = int(user_motion.size)
        fallback_reason = "both_constant"
    elif user_stats.constant_signal or idol_stats.constant_signal:
        lag, score = 0, 0.0
        direction = "unobservable"
        overlap_count = int(user_motion.size)
        fallback_reason = "one_constant"
    else:
        lag, correlation, overlap_count = _select_best_lag(
            user_stats.signal, idol_stats.signal, active_config
        )
        score = clip_score(
            100.0 - abs(lag) * active_config.timing_penalty_per_frame
        )
        direction = (
            "aligned" if lag == 0 else "user_delayed" if lag > 0 else "user_ahead"
        )
        reliable = correlation > 0.0

    aligned_user, aligned_idol = align_pose_sequences(
        user, idol, lag, active_config
    )
    lag_ms = float(lag / active_config.fps * 1000.0)
    diagnostics = TimingDiagnostics(
        lag_frames=int(lag),
        lag_ms=lag_ms,
        abs_lag_ms=float(abs(lag) / active_config.fps * 1000.0),
        lag_direction=direction,
        correlation_at_best_lag=correlation,
        max_lag_frames=int(active_config.max_lag_frames),
        penalty_per_frame=float(active_config.timing_penalty_per_frame),
        original_frame_count=int(user.shape[0]),
        aligned_frame_count=int(aligned_user.shape[0]),
        motion_overlap_count=int(overlap_count),
        user_motion_std=float(user_stats.std),
        idol_motion_std=float(idol_stats.std),
        user_constant_signal=bool(user_stats.constant_signal),
        idol_constant_signal=bool(idol_stats.constant_signal),
        reliable=bool(reliable),
        fallback_reason=fallback_reason,
    )
    return TimingScoreResult(
        score=float(score),
        aligned_user_seq=aligned_user,
        aligned_idol_seq=aligned_idol,
        diagnostics=diagnostics,
    )
