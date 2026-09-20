"""Integrated rule-only pentagon scoring pipeline."""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    fields,
    is_dataclass,
)
import math
from numbers import Real
from typing import Any

import numpy as np

from .pentagon_common import clip_score
from .pentagon_config import PentagonConfig
from .score_accuracy import compute_accuracy_score
from .score_balance import compute_balance_score
from .score_detail import compute_detail_score
from .score_rhythm import compute_rhythm_score
from .score_timing import compute_timing_score


PENTAGON_SCORE_NAMES: tuple[str, ...] = (
    "accuracy",
    "detail",
    "balance",
    "timing",
    "rhythm",
)

PENTAGON_SCORE_WEIGHTS: dict[str, float] = {
    "accuracy": 0.30,
    "detail": 0.15,
    "balance": 0.20,
    "timing": 0.15,
    "rhythm": 0.20,
}


@dataclass(frozen=True)
class PentagonScores:
    accuracy: float
    detail: float
    balance: float
    timing: float
    rhythm: float

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(
                getattr(self, name)
            )
            for name in PENTAGON_SCORE_NAMES
        }

    def values_in_order(
        self,
    ) -> tuple[float, ...]:
        return tuple(
            float(
                getattr(self, name)
            )
            for name in PENTAGON_SCORE_NAMES
        )


@dataclass(frozen=True)
class PentagonWeights:
    accuracy: float
    detail: float
    balance: float
    timing: float
    rhythm: float

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(
                getattr(self, name)
            )
            for name in PENTAGON_SCORE_NAMES
        }

    def values_in_order(
        self,
    ) -> tuple[float, ...]:
        return tuple(
            float(
                getattr(self, name)
            )
            for name in PENTAGON_SCORE_NAMES
        )




@dataclass(frozen=True)
class PentagonReliability:
    timing: bool
    accuracy: bool
    detail: bool
    balance: bool
    rhythm: bool
    overall: bool
    unreliable_components: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timing": bool(self.timing),
            "accuracy": bool(self.accuracy),
            "detail": bool(self.detail),
            "balance": bool(self.balance),
            "rhythm": bool(self.rhythm),
            "overall": bool(self.overall),
            "unreliable_components": list(
                self.unreliable_components
            ),
        }


@dataclass(frozen=True)
class PentagonScoreResult:
    final_score: float
    rule_score: float
    dca_score: float | None
    dca_used_in_final_score: bool
    score_mode: str
    scores: PentagonScores
    weights: PentagonWeights
    reliability: PentagonReliability
    input_frame_count: int
    aligned_frame_count: int
    lag_frames: int
    lag_ms: float
    component_results: dict[str, Any]
    feedback_summary: dict[str, Any]
    pipeline_version: str

    def to_dict(self) -> dict[str, Any]:
        return _to_json_compatible(
            asdict(self)
        )


def _validate_top_k(
    top_k: int,
) -> int:
    if (
        isinstance(top_k, bool)
        or not isinstance(
            top_k,
            (int, np.integer),
        )
    ):
        raise ValueError(
            "top_k must be an integer, not bool; "
            f"got {top_k!r}"
        )

    value = int(top_k)

    if value < 1 or value > 5:
        raise ValueError(
            "top_k must be in [1, 5]; "
            f"got {value}"
        )

    return value


def _validate_optional_dca_score(
    value: Any,
) -> float | None:
    if value is None:
        return None

    if (
        isinstance(
            value,
            (bool, np.bool_),
        )
        or not isinstance(
            value,
            Real,
        )
    ):
        raise ValueError(
            "dca_score must be a number "
            f"in [0, 100]; got {value!r}"
        )

    result = float(value)

    if (
        not math.isfinite(result)
        or result < 0.0
        or result > 100.0
    ):
        raise ValueError(
            "dca_score must be finite and "
            f"in [0, 100]; got {value!r}"
        )

    return result


def _validate_score_value(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(
            value,
            (bool, np.bool_),
        )
        or not isinstance(
            value,
            Real,
        )
    ):
        raise ValueError(
            f"{name} score must be numeric; "
            f"got {value!r}"
        )

    result = float(value)

    if (
        not math.isfinite(result)
        or result < 0.0
        or result > 100.0
    ):
        raise ValueError(
            f"{name} score must be finite "
            f"and in [0, 100]; got {value!r}"
        )

    return result


def _validate_weights(
    weights: dict[str, Any],
) -> dict[str, float]:
    if (
        set(weights)
        != set(PENTAGON_SCORE_NAMES)
    ):
        raise ValueError(
            "pentagon weight names must match "
            "PENTAGON_SCORE_NAMES"
        )

    validated: dict[str, float] = {}

    for name in PENTAGON_SCORE_NAMES:
        value = weights[name]

        if (
            isinstance(
                value,
                (bool, np.bool_),
            )
            or not isinstance(
                value,
                Real,
            )
        ):
            raise ValueError(
                f"weight {name} must be numeric"
            )

        number = float(value)

        if (
            not math.isfinite(number)
            or number < 0.0
        ):
            raise ValueError(
                f"weight {name} must be "
                "finite and nonnegative"
            )

        validated[name] = number

    weight_sum = math.fsum(
        validated.values()
    )

    if not math.isclose(
        weight_sum,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "pentagon weights must sum to 1.0"
        )

    return validated


def _extract_reliability(
    result: Any,
) -> bool:
    return bool(
        getattr(
            result.diagnostics,
            "reliable",
            True,
        )
    )


def _to_json_compatible(
    value: Any,
) -> Any:
    if (
        is_dataclass(value)
        and not isinstance(value, type)
    ):
        return {
            field.name: _to_json_compatible(
                getattr(
                    value,
                    field.name,
                )
            )
            for field in fields(value)
        }

    if isinstance(value, dict):
        return {
            str(key): _to_json_compatible(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _to_json_compatible(item)
            for item in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return [
            _to_json_compatible(item)
            for item in value.tolist()
        ]

    if isinstance(
        value,
        np.generic,
    ):
        return _to_json_compatible(
            value.item()
        )

    if (
        value is None
        or isinstance(
            value,
            (
                str,
                bool,
                int,
                float,
            ),
        )
    ):
        return value

    raise ValueError(
        "value is not JSON-compatible: "
        f"{type(value).__name__}"
    )


def compute_pentagon_scores(
    user_seq: Any,
    idol_seq: Any,
    config: PentagonConfig | None = None,
    dca_score: Any = None,
    top_k: int = 5,
) -> PentagonScoreResult:
    """Run Timing once, then score its aligned normalized pose pair."""

    active_config = (
        PentagonConfig()
        if config is None
        else config
    )

    active_config.validate()

    selected_top_k = _validate_top_k(
        top_k
    )

    validated_dca = (
        _validate_optional_dca_score(
            dca_score
        )
    )

    weights_dict = _validate_weights(
        PENTAGON_SCORE_WEIGHTS
    )

    # Timing에서 정규화와 lag 탐색을 한 번만 수행한다.
    timing_result = compute_timing_score(
        user_seq,
        idol_seq,
        active_config,
    )

    aligned_user = (
        timing_result.aligned_user_seq
    )

    aligned_idol = (
        timing_result.aligned_idol_seq
    )

    accuracy_result = (
        compute_accuracy_score(
            aligned_user,
            aligned_idol,
            active_config,
            top_k=selected_top_k,
        )
    )

    detail_result = (
        compute_detail_score(
            aligned_user,
            aligned_idol,
            active_config,
            top_k=selected_top_k,
        )
    )

    balance_result = (
        compute_balance_score(
            aligned_user,
            aligned_idol,
            active_config,
            top_k=selected_top_k,
        )
    )

    rhythm_result = (
        compute_rhythm_score(
            aligned_user,
            aligned_idol,
            active_config,
        )
    )

    result_map = {
        "accuracy": accuracy_result,
        "detail": detail_result,
        "balance": balance_result,
        "timing": timing_result,
        "rhythm": rhythm_result,
    }

    score_values = {
        name: _validate_score_value(
            result_map[name].score,
            name,
        )
        for name in PENTAGON_SCORE_NAMES
    }

    scores = PentagonScores(
        **score_values
    )

    weights = PentagonWeights(
        **weights_dict
    )

    rule_score = float(
        clip_score(
            math.fsum(
                score_values[name]
                * weights_dict[name]
                for name
                in PENTAGON_SCORE_NAMES
            )
        )
    )

    execution_order = (
        "timing",
        "accuracy",
        "detail",
        "balance",
        "rhythm",
    )

    reliability_values = {
        name: _extract_reliability(
            result_map[name]
        )
        for name in execution_order
    }

    unreliable_components = [
        name
        for name in execution_order
        if not reliability_values[name]
    ]

    reliability = PentagonReliability(
        **reliability_values,
        overall=all(
            reliability_values.values()
        ),
        unreliable_components=(
            unreliable_components
        ),
    )

    component_results = {
        "timing": (
            timing_result.to_dict()
        ),
        "accuracy": (
            accuracy_result.to_dict()
        ),
        "detail": (
            detail_result.to_dict()
        ),
        "balance": (
            balance_result.to_dict()
        ),
        "rhythm": (
            rhythm_result.to_dict()
        ),
    }

    accuracy_diagnostics = (
        accuracy_result.diagnostics
    )

    detail_diagnostics = (
        detail_result.diagnostics
    )

    balance_diagnostics = (
        balance_result.diagnostics
    )

    rhythm_diagnostics = (
        rhythm_result.diagnostics
    )

    timing_diagnostics = (
        timing_result.diagnostics
    )

    worst_joint = (
        accuracy_diagnostics
        .worst_joints[0]
        if accuracy_diagnostics.worst_joints
        else None
    )

    feedback_summary = {
        "timing": {
            "lag_frames": int(
                timing_diagnostics
                .lag_frames
            ),
            "lag_ms": float(
                timing_diagnostics
                .lag_ms
            ),
        },
        "accuracy": {
            "worst_joint": worst_joint,
            "worst_joint_index": (
                None
                if worst_joint is None
                else int(
                    worst_joint["index"]
                )
            ),
            "worst_joint_name": (
                None
                if worst_joint is None
                else str(
                    worst_joint["name"]
                )
            ),
            "weighted_contribution": (
                None
                if worst_joint is None
                else float(
                    worst_joint[
                        "weighted_contribution"
                    ]
                )
            ),
        },
        "detail": {
            "worst_position_joints": (
                detail_diagnostics
                .worst_position_joints
            ),
            "worst_angles": (
                detail_diagnostics
                .worst_angles
            ),
            "fallback_reason": (
                detail_diagnostics
                .fallback_reason
            ),
        },
        "balance": {
            "worst_components": (
                balance_diagnostics
                .worst_components
            ),
            "fallback_reason": (
                balance_diagnostics
                .fallback_reason
            ),
            "camera_assumption": (
                balance_diagnostics
                .camera_assumption
            ),
        },
        "rhythm": {
            "motion_correlation": (
                rhythm_diagnostics
                .motion_correlation
            ),
            "normalized_motion_mae": float(
                rhythm_diagnostics
                .normalized_motion_mae
            ),
            "fallback_reason": (
                rhythm_diagnostics
                .fallback_reason
            ),
        },
    }

    return PentagonScoreResult(
        final_score=rule_score,
        rule_score=rule_score,
        dca_score=validated_dca,
        dca_used_in_final_score=False,
        score_mode="rule_only",
        scores=scores,
        weights=weights,
        reliability=reliability,
        input_frame_count=int(
            timing_diagnostics
            .original_frame_count
        ),
        aligned_frame_count=int(
            timing_diagnostics
            .aligned_frame_count
        ),
        lag_frames=int(
            timing_diagnostics
            .lag_frames
        ),
        lag_ms=float(
            timing_diagnostics
            .lag_ms
        ),
        component_results=(
            _to_json_compatible(
                component_results
            )
        ),
        feedback_summary=(
            _to_json_compatible(
                feedback_summary
            )
        ),
        pipeline_version=(
            "pentagon_rule_v1"
        ),
    )
