"""Configuration and MediaPipe Pose indices for pentagon scoring foundations.

The constants in this module are explicit MVP candidates.  They have not been
calibrated against real dance videos or human ratings and require real-data
calibration before production use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class MediaPipePoseIndices:
    """MediaPipe Pose landmark indices used by the rule-based pipeline."""

    left_shoulder: int = 11
    right_shoulder: int = 12
    left_elbow: int = 13
    right_elbow: int = 14
    left_wrist: int = 15
    right_wrist: int = 16
    left_pinky: int = 17
    right_pinky: int = 18
    left_index: int = 19
    right_index: int = 20
    left_thumb: int = 21
    right_thumb: int = 22
    left_hip: int = 23
    right_hip: int = 24
    left_knee: int = 25
    right_knee: int = 26
    left_ankle: int = 27
    right_ankle: int = 28
    left_heel: int = 29
    right_heel: int = 30
    left_foot_index: int = 31
    right_foot_index: int = 32


MEDIAPIPE_POSE = MediaPipePoseIndices()

# Face landmarks receive less weight than the torso and major limb joints.
# These immutable values are MVP candidates and require real-data calibration.
ACCURACY_JOINT_WEIGHTS: tuple[float, ...] = (
    0.35, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.25, 0.25, 0.25, 0.25,
    1.50, 1.50, 1.25, 1.25, 1.10, 1.10, 0.55, 0.55, 0.60, 0.60, 0.55,
    0.55, 1.60, 1.60, 1.35, 1.35, 1.25, 1.25, 0.85, 0.85, 1.10, 1.10,
)

@dataclass(frozen=True)
class PentagonConfig:
    """Validated settings for common pose operations.

    This module assumes the third channel is z as an input contract.  The
    caller must guarantee that the actual input is ``[x, y, z]``.  This
    contract is not evidence that the existing DCA prototype was trained with
    real MediaPipe z values.

    All scales and weights are MVP candidates that require calibration using
    real videos and human ratings.
    """

    expected_frames: int | None = 30
    num_joints: int = 33
    coord_dim: int = 3
    fps: float = 30.0
    use_z: bool = True
    z_weight: float = 0.3
    epsilon: float = 1e-8
    min_shoulder_width: float = 1e-6
    max_lag_frames: int = 5
    minimum_aligned_frames: int = 15
    timing_penalty_per_frame: float = 5.0
    accuracy_scale: float = 0.3
    rhythm_mae_scale: float = 25.0

    rhythm_correlation_weight: float = 0.60
    rhythm_mae_weight: float = 0.40

    accuracy_joint_weights: tuple[float, ...] = field(
        default=ACCURACY_JOINT_WEIGHTS
    )
    def __post_init__(self) -> None:
        """Validate the configuration immediately after construction."""
        self.validate()

    def validate(self) -> None:
        """Validate dimensions, ranges, indices, and all weight group sums.

        Raises:
            ValueError: If a dimension/range is invalid, a weight is negative,
                an index is out of range, or a weight group does not sum to 1.
        """
        if self.fps <= 0:
            raise ValueError(f"fps must be > 0; got {self.fps}")
        if self.expected_frames is not None and self.expected_frames < 2:
            raise ValueError(
                "expected_frames must be None or >= 2; "
                f"got {self.expected_frames}"
            )
        if self.num_joints != 33:
            raise ValueError(f"num_joints must be 33; got {self.num_joints}")
        if self.coord_dim != 3:
            raise ValueError(f"coord_dim must be 3; got {self.coord_dim}")
        if self.z_weight < 0:
            raise ValueError(f"z_weight must be >= 0; got {self.z_weight}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0; got {self.epsilon}")
        if self.min_shoulder_width <= 0:
            raise ValueError(
                "min_shoulder_width must be > 0; "
                f"got {self.min_shoulder_width}"
            )
        if self.max_lag_frames < 0:
            raise ValueError(
                f"max_lag_frames must be >= 0; got {self.max_lag_frames}"
            )
        if self.minimum_aligned_frames < 2:
            raise ValueError(
                "minimum_aligned_frames must be >= 2; "
                f"got {self.minimum_aligned_frames}"
            )

        named_weights = {
            "rhythm_correlation_weight": self.rhythm_correlation_weight,
            "rhythm_mae_weight": self.rhythm_mae_weight,
        }
        for name, value in named_weights.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0; got {value}")

        groups = {
            "rhythm": (
                self.rhythm_correlation_weight,
                self.rhythm_mae_weight,
            ),
        }
        for group_name, values in groups.items():
            total = math.fsum(values)
            if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"{group_name} weights must sum to 1.0; got {total}"
                )

        if len(self.accuracy_joint_weights) != self.num_joints:
            raise ValueError(
                "accuracy_joint_weights must contain 33 values; "
                f"got {len(self.accuracy_joint_weights)}"
            )
        if any(weight < 0 for weight in self.accuracy_joint_weights):
            raise ValueError("accuracy_joint_weights must all be >= 0")
        if not any(weight > 0 for weight in self.accuracy_joint_weights):
            raise ValueError("accuracy_joint_weights must contain a positive value")
