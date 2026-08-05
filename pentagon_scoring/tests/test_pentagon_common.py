"""Unit tests for the 11-2 pentagon common foundation."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from pentagon_common import (
    clip_score,
    compute_hip_center,
    compute_joint_angles,
    compute_motion_signal,
    compute_weighted_joint_distances,
    normalize_pose_sequence,
    normalize_signal,
    prepare_pose_pair,
    validate_pose_sequence,
)
from pentagon_config import PentagonConfig


def make_skeleton(frames: int = 30) -> np.ndarray:
    """Build a deterministic, meaningful 33-joint frontal pose sequence."""
    pose = np.zeros((frames, 33, 3), dtype=np.float64)
    for frame in range(frames):
        center_x = 0.03 * frame
        center_y = 0.01 * frame
        center_z = 0.005 * frame
        pose[frame, :, 0] = center_x
        pose[frame, :, 1] = center_y
        pose[frame, :, 2] = center_z

        # Torso and major limb landmarks.
        pose[frame, 11] += (-1.0, -2.0, 0.10)
        pose[frame, 12] += (1.0, -2.0, -0.10)
        pose[frame, 13] += (-1.5, -1.0, 0.05)
        pose[frame, 14] += (1.5, -1.0, -0.05)
        pose[frame, 15] += (-2.0, 0.0, 0.00)
        pose[frame, 16] += (2.0, 0.0, 0.00)
        pose[frame, 23] += (-0.4, 0.0, 0.05)
        pose[frame, 24] += (0.4, 0.0, -0.05)
        pose[frame, 25] += (-0.5, 1.5, 0.02)
        pose[frame, 26] += (0.5, 1.5, -0.02)
        pose[frame, 27] += (-0.6, 3.0, 0.00)
        pose[frame, 28] += (0.6, 3.0, 0.00)
        pose[frame, 29] += (-0.7, 3.2, -0.10)
        pose[frame, 30] += (0.7, 3.2, -0.10)
        pose[frame, 31] += (-0.8, 3.3, 0.20)
        pose[frame, 32] += (0.8, 3.3, 0.20)

        # Limited MediaPipe hand endpoints near each wrist.
        pose[frame, 17] += (-2.10, 0.05, 0.00)
        pose[frame, 18] += (2.10, 0.05, 0.00)
        pose[frame, 19] += (-2.15, 0.00, 0.00)
        pose[frame, 20] += (2.15, 0.00, 0.00)
        pose[frame, 21] += (-2.05, -0.05, 0.00)
        pose[frame, 22] += (2.05, -0.05, 0.00)
    return pose


class ValidatePoseSequenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PentagonConfig()
        self.pose = make_skeleton()

    def test_valid_shape(self) -> None:
        result = validate_pose_sequence(self.pose, self.config)
        self.assertEqual(result.shape, (30, 33, 3))

    def test_wrong_ndim(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape="):
            validate_pose_sequence(np.zeros((33, 3)), self.config, "user_seq")

    def test_wrong_joint_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "user_seq.*shape="):
            validate_pose_sequence(
                np.zeros((30, 32, 3)), self.config, "user_seq"
            )

    def test_wrong_coordinate_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "idol_seq.*shape="):
            validate_pose_sequence(
                np.zeros((30, 33, 2)), self.config, "idol_seq"
            )

    def test_wrong_frame_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "30 frames.*shape=\\(29, 33, 3\\)"):
            validate_pose_sequence(np.zeros((29, 33, 3)), self.config)

    def test_nan_rejected(self) -> None:
        self.pose[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN.*shape="):
            validate_pose_sequence(self.pose, self.config)

    def test_inf_rejected(self) -> None:
        self.pose[0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "Inf.*shape="):
            validate_pose_sequence(self.pose, self.config)

    def test_float32_conversion(self) -> None:
        result = validate_pose_sequence(self.pose, self.config)
        self.assertEqual(result.dtype, np.float32)

    def test_c_contiguous_conversion(self) -> None:
        noncontiguous = self.pose[:, ::-1, :]
        result = validate_pose_sequence(noncontiguous, self.config)
        self.assertTrue(result.flags.c_contiguous)


class NormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PentagonConfig()
        self.pose = make_skeleton()

    def test_hip_center_is_zero_after_normalization(self) -> None:
        normalized = normalize_pose_sequence(self.pose, self.config)
        centers = compute_hip_center(normalized)
        np.testing.assert_allclose(centers, 0.0, atol=1e-6)

    def test_median_shoulder_width_is_one(self) -> None:
        _, diagnostics = normalize_pose_sequence(
            self.pose, self.config, return_diagnostics=True
        )
        self.assertAlmostEqual(
            diagnostics.median_shoulder_width_after, 1.0, places=6
        )

    def test_translation_invariance(self) -> None:
        translation = np.array([8.0, -3.0, 2.5])
        first = normalize_pose_sequence(self.pose, self.config)
        second = normalize_pose_sequence(
            self.pose + translation, self.config
        )
        np.testing.assert_allclose(first, second, atol=2e-6)

    def test_positive_scale_invariance(self) -> None:
        first = normalize_pose_sequence(self.pose, self.config)
        second = normalize_pose_sequence(self.pose * 3.5, self.config)
        np.testing.assert_allclose(first, second, atol=2e-6)

    def test_identical_pair_remains_identical(self) -> None:
        user, idol = prepare_pose_pair(self.pose, self.pose.copy(), self.config)
        np.testing.assert_array_equal(user, idol)

    def test_pair_diagnostics_are_separate(self) -> None:
        _, _, diagnostics = prepare_pose_pair(
            self.pose,
            self.pose * 2.0,
            self.config,
            return_diagnostics=True,
        )
        self.assertIsNotNone(diagnostics.user)
        self.assertIsNotNone(diagnostics.idol)
        self.assertNotEqual(
            diagnostics.user.scale_factor, diagnostics.idol.scale_factor
        )

    def test_zero_shoulder_width_rejected(self) -> None:
        pose = self.pose.copy()
        pose[:, 12, :2] = pose[:, 11, :2]
        with self.assertRaisesRegex(ValueError, "no valid xy shoulder widths"):
            normalize_pose_sequence(pose, self.config)

    def test_tiny_shoulder_width_rejected(self) -> None:
        pose = self.pose.copy()
        pose[:, 12, :2] = pose[:, 11, :2]
        pose[:, 12, 0] += self.config.min_shoulder_width / 2
        with self.assertRaisesRegex(ValueError, "no valid xy shoulder widths"):
            normalize_pose_sequence(pose, self.config)


class MathHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PentagonConfig()
        self.pose = make_skeleton()

    def test_identical_weighted_joint_distances_are_zero(self) -> None:
        distances = compute_weighted_joint_distances(
            self.pose, self.pose.copy(), self.config
        )
        np.testing.assert_array_equal(distances, 0.0)

    def test_z_difference_counts_when_enabled(self) -> None:
        other = self.pose.copy()
        other[:, 15, 2] += 2.0
        distances = compute_weighted_joint_distances(
            self.pose, other, self.config, joint_indices=(15,)
        )
        self.assertTrue(np.all(distances > 0))

    def test_z_difference_ignored_when_disabled(self) -> None:
        other = self.pose.copy()
        other[:, 15, 2] += 2.0
        config = replace(self.config, use_z=False)
        distances = compute_weighted_joint_distances(
            self.pose, other, config, joint_indices=(15,)
        )
        np.testing.assert_array_equal(distances, 0.0)

    def test_straight_triplet_is_approximately_180_degrees(self) -> None:
        pose = self.pose.copy()
        pose[:, 0] = (-1.0, 0.0, 0.0)
        pose[:, 1] = (0.0, 0.0, 0.0)
        pose[:, 2] = (1.0, 0.0, 0.0)
        result = compute_joint_angles(pose, ((0, 1, 2),), self.config)
        self.assertTrue(np.all(result.valid_mask))
        np.testing.assert_allclose(result.angles, 180.0, atol=0.02)

    def test_right_angle_is_approximately_90_degrees(self) -> None:
        pose = self.pose.copy()
        pose[:, 0] = (1.0, 0.0, 0.0)
        pose[:, 1] = (0.0, 0.0, 0.0)
        pose[:, 2] = (0.0, 1.0, 0.0)
        result = compute_joint_angles(pose, ((0, 1, 2),), self.config)
        np.testing.assert_allclose(result.angles, 90.0, atol=0.02)

    def test_degenerate_angle_has_nan_and_false_mask(self) -> None:
        pose = self.pose.copy()
        pose[:, 0] = pose[:, 1]
        result = compute_joint_angles(pose, ((0, 1, 2),), self.config)
        self.assertTrue(np.isnan(result.angles).all())
        self.assertFalse(result.valid_mask.any())

    def test_stationary_sequence_motion_is_zero(self) -> None:
        pose = np.repeat(self.pose[:1], 30, axis=0)
        signal = compute_motion_signal(pose, self.config)
        self.assertEqual(signal.shape, (29,))
        np.testing.assert_array_equal(signal, 0.0)

    def test_constant_signal_normalizes_to_zero(self) -> None:
        result = normalize_signal(np.full(29, 4.0), self.config)
        self.assertTrue(result.constant_signal)
        np.testing.assert_array_equal(result.signal, 0.0)

    def test_nonconstant_signal_reports_flag(self) -> None:
        result = normalize_signal(np.arange(5), self.config)
        self.assertFalse(result.constant_signal)
        self.assertAlmostEqual(float(result.signal.mean()), 0.0, places=6)

    def test_clip_score_bounds(self) -> None:
        self.assertEqual(clip_score(-2.0), 0.0)
        self.assertEqual(clip_score(44.5), 44.5)
        self.assertEqual(clip_score(120.0), 100.0)


class ConfigValidationTests(unittest.TestCase):
    def test_rhythm_weights_sum_to_one(self) -> None:
        config = PentagonConfig()
        self.assertAlmostEqual(
            config.rhythm_correlation_weight + config.rhythm_mae_weight, 1.0
        )
        with self.assertRaisesRegex(ValueError, "rhythm weights"):
            replace(config, rhythm_correlation_weight=0.61)

    def test_negative_weight_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be >= 0"):
            PentagonConfig(
                rhythm_correlation_weight=-0.1,
                rhythm_mae_weight=1.1,
            )

    def test_invalid_dimensions_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_joints must be 33"):
            PentagonConfig(num_joints=32)
        with self.assertRaisesRegex(ValueError, "coord_dim must be 3"):
            PentagonConfig(coord_dim=2)


if __name__ == "__main__":
    unittest.main()
