"""Integration tests for the rule-only pentagon score pipeline."""

from __future__ import annotations

from dataclasses import replace
import json
import unittest

import numpy as np

from app.pentagon.pentagon_config import PentagonConfig
from app.pentagon.pentagon_scores import (
    PENTAGON_SCORE_NAMES,
    PENTAGON_SCORE_WEIGHTS,
    PentagonReliability,
    PentagonScoreResult,
    PentagonScores,
    PentagonWeights,
    _to_json_compatible,
    _validate_score_value,
    _validate_weights,
    compute_pentagon_scores,
)


PATTERN = np.array(
    [
        0, 0, 1, 3, 1,
        0, 0, -2, -2, 1,
        4, 1, 0, 0, 2,
        5, 2, -1, -1, 0,
        3, 3, 0, -3, 0,
        1, 4, 0, -2, 0,
    ],
    dtype=np.float64,
)


def make_pose(
    pattern: np.ndarray = PATTERN,
) -> np.ndarray:
    values = np.asarray(
        pattern,
        dtype=np.float64,
    )

    pose = np.zeros(
        (30, 33, 3),
        dtype=np.float64,
    )

    pose[:, 11, :2] = (-0.5, -1.0)
    pose[:, 12, :2] = (0.5, -1.0)

    pose[:, 13, :2] = (-0.8, -0.4)
    pose[:, 14, :2] = (0.8, -0.4)

    pose[:, 15, :2] = (-1.0, 0.1)
    pose[:, 16, :2] = (1.0, 0.1)

    pose[:, 23, :2] = (-0.25, 0.0)
    pose[:, 24, :2] = (0.25, 0.0)

    pose[:, 25, :2] = (-0.30, 0.5)
    pose[:, 26, :2] = (0.30, 0.5)

    pose[:, 27, :2] = (-0.35, 1.0)
    pose[:, 28, :2] = (0.35, 1.0)

    pose[:, 13, 0] += 0.04 * values
    pose[:, 14, 0] -= 0.03 * values

    pose[:, 15, 1] += 0.12 * values
    pose[:, 16, 1] -= 0.08 * values

    pose[:, 27, 0] += 0.05 * values
    pose[:, 28, 0] -= 0.06 * values

    return pose


def delayed(
    sequence: np.ndarray,
    lag: int = 3,
) -> np.ndarray:
    result = np.empty_like(
        sequence
    )

    result[:lag] = sequence[0]
    result[lag:] = sequence[:-lag]

    return result


def contains_numpy(
    value,
) -> bool:
    if isinstance(
        value,
        (np.ndarray, np.generic),
    ):
        return True

    if isinstance(value, dict):
        return any(
            contains_numpy(key)
            or contains_numpy(item)
            for key, item
            in value.items()
        )

    if isinstance(
        value,
        (list, tuple),
    ):
        return any(
            contains_numpy(item)
            for item in value
        )

    return False


def contains_key(
    value,
    target: str,
) -> bool:
    if isinstance(value, dict):
        return (
            target in value
            or any(
                contains_key(
                    item,
                    target,
                )
                for item
                in value.values()
            )
        )

    if isinstance(
        value,
        (list, tuple),
    ):
        return any(
            contains_key(
                item,
                target,
            )
            for item in value
        )

    return False


class PentagonIntegrationTests(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls) -> None:
        cls.pose = make_pose()

        cls.identical = (
            compute_pentagon_scores(
                cls.pose,
                cls.pose,
            )
        )

    def test_01_identical_five_scores_100(
        self,
    ):
        self.assertEqual(
            self.identical
            .scores
            .values_in_order(),
            (100.0,) * 5,
        )

    def test_02_identical_rule_100(
        self,
    ):
        self.assertEqual(
            self.identical.rule_score,
            100.0,
        )

    def test_03_identical_final_100(
        self,
    ):
        self.assertEqual(
            self.identical.final_score,
            100.0,
        )

    def test_04_final_equals_rule(
        self,
    ):
        self.assertEqual(
            self.identical.final_score,
            self.identical.rule_score,
        )

    def test_05_score_mode(
        self,
    ):
        self.assertEqual(
            self.identical.score_mode,
            "rule_only",
        )

    def test_06_pipeline_version(
        self,
    ):
        self.assertEqual(
            self.identical.pipeline_version,
            "pentagon_rule_v1",
        )

    def test_07_default_weights(
        self,
    ):
        self.assertEqual(
            self.identical.weights.to_dict(),
            PENTAGON_SCORE_WEIGHTS,
        )

    def test_08_weight_sum(
        self,
    ):
        self.assertAlmostEqual(
            sum(
                self.identical
                .weights
                .values_in_order()
            ),
            1.0,
        )

    def test_09_score_order(
        self,
    ):
        expected = tuple(
            getattr(
                self.identical.scores,
                name,
            )
            for name
            in PENTAGON_SCORE_NAMES
        )

        self.assertEqual(
            self.identical
            .scores
            .values_in_order(),
            expected,
        )

    def test_10_weight_order(
        self,
    ):
        expected = tuple(
            PENTAGON_SCORE_WEIGHTS[
                name
            ]
            for name
            in PENTAGON_SCORE_NAMES
        )

        self.assertEqual(
            self.identical
            .weights
            .values_in_order(),
            expected,
        )

    def test_11_rule_formula(
        self,
    ):
        expected = sum(
            getattr(
                self.identical.scores,
                name,
            )
            * PENTAGON_SCORE_WEIGHTS[
                name
            ]
            for name
            in PENTAGON_SCORE_NAMES
        )

        self.assertAlmostEqual(
            self.identical.rule_score,
            expected,
        )

    def test_12_component_ranges(
        self,
    ):
        self.assertTrue(
            all(
                0.0 <= value <= 100.0
                for value
                in self.identical
                .scores
                .values_in_order()
            )
        )

    def test_13_final_python_float(
        self,
    ):
        self.assertIs(
            type(
                self.identical.final_score
            ),
            float,
        )

    def test_14_rule_python_float(
        self,
    ):
        self.assertIs(
            type(
                self.identical.rule_score
            ),
            float,
        )

    def test_15_dca_default_none(
        self,
    ):
        self.assertIsNone(
            self.identical.dca_score
        )

    def test_16_dca_not_used(
        self,
    ):
        self.assertFalse(
            self.identical
            .dca_used_in_final_score
        )

    def test_17_dca_passthrough(
        self,
    ):
        result = compute_pentagon_scores(
            self.pose,
            self.pose,
            dca_score=80,
        )

        self.assertEqual(
            result.dca_score,
            80.0,
        )

    def test_18_dca_does_not_change_final(
        self,
    ):
        result = compute_pentagon_scores(
            self.pose,
            self.pose,
            dca_score=80,
        )

        self.assertEqual(
            result.final_score,
            self.identical.final_score,
        )

    def test_19_negative_dca(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            compute_pentagon_scores(
                self.pose,
                self.pose,
                dca_score=-1,
            )

    def test_20_high_dca(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            compute_pentagon_scores(
                self.pose,
                self.pose,
                dca_score=101,
            )

    def test_21_nan_dca(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            compute_pentagon_scores(
                self.pose,
                self.pose,
                dca_score=np.nan,
            )

    def test_22_inf_dca(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            compute_pentagon_scores(
                self.pose,
                self.pose,
                dca_score=np.inf,
            )

    def test_23_bool_dca(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            compute_pentagon_scores(
                self.pose,
                self.pose,
                dca_score=True,
            )

    def test_24_top_k_one(
        self,
    ):
        result = compute_pentagon_scores(
            self.pose,
            self.pose,
            top_k=1,
        )

        self.assertEqual(
            result.component_results[
                "accuracy"
            ]["diagnostics"]["top_k"],
            1,
        )

    def test_25_top_k_five(
        self,
    ):
        self.assertEqual(
            self.identical
            .component_results[
                "balance"
            ]["diagnostics"]["top_k"],
            5,
        )

    def test_26_top_k_zero(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            compute_pentagon_scores(
                self.pose,
                self.pose,
                top_k=0,
            )

    def test_27_top_k_six(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            compute_pentagon_scores(
                self.pose,
                self.pose,
                top_k=6,
            )

    def test_28_top_k_bool(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            compute_pentagon_scores(
                self.pose,
                self.pose,
                top_k=True,
            )

    def test_29_input_frames(
        self,
    ):
        self.assertEqual(
            self.identical.input_frame_count,
            30,
        )

    def test_30_aligned_frames(
        self,
    ):
        self.assertEqual(
            self.identical
            .aligned_frame_count,
            30,
        )

    def test_31_lag_frames(
        self,
    ):
        self.assertEqual(
            self.identical.lag_frames,
            0,
        )

    def test_32_lag_ms(
        self,
    ):
        self.assertEqual(
            self.identical.lag_ms,
            0.0,
        )

    def test_33_component_keys(
        self,
    ):
        self.assertEqual(
            set(
                self.identical
                .component_results
            ),
            {
                "timing",
                "accuracy",
                "detail",
                "balance",
                "rhythm",
            },
        )

    def test_34_component_scores_match(
        self,
    ):
        for name in (
            PENTAGON_SCORE_NAMES
        ):
            self.assertEqual(
                self.identical
                .component_results[
                    name
                ]["score"],
                getattr(
                    self.identical.scores,
                    name,
                ),
            )

    def test_35_component_diagnostics(
        self,
    ):
        self.assertTrue(
            all(
                "diagnostics" in value
                for value
                in self.identical
                .component_results
                .values()
            )
        )

    def test_36_json_serializable(
        self,
    ):
        self.assertIsInstance(
            json.dumps(
                self.identical.to_dict()
            ),
            str,
        )

    def test_37_no_ndarray(
        self,
    ):
        self.assertFalse(
            contains_numpy(
                self.identical.to_dict()
            )
        )

    def test_38_no_numpy_scalar(
        self,
    ):
        self.assertFalse(
            contains_numpy(
                self.identical.to_dict()
            )
        )

    def test_39_no_aligned_user(
        self,
    ):
        self.assertFalse(
            contains_key(
                self.identical.to_dict(),
                "aligned_user_seq",
            )
        )

    def test_40_no_aligned_idol(
        self,
    ):
        self.assertFalse(
            contains_key(
                self.identical.to_dict(),
                "aligned_idol_seq",
            )
        )

    def test_41_inputs_unchanged(
        self,
    ):
        user = self.pose.copy()
        idol = self.pose.copy()

        user_before = user.copy()
        idol_before = idol.copy()

        compute_pentagon_scores(
            user,
            idol,
        )

        np.testing.assert_array_equal(
            user,
            user_before,
        )

        np.testing.assert_array_equal(
            idol,
            idol_before,
        )

    def test_42_config_unchanged(
        self,
    ):
        config = PentagonConfig()

        compute_pentagon_scores(
            self.pose,
            self.pose,
            config,
        )

        self.assertEqual(
            config.expected_frames,
            30,
        )

    def test_43_deterministic(
        self,
    ):
        current = compute_pentagon_scores(
            self.pose,
            self.pose,
        )

        self.assertEqual(
            current.to_dict(),
            self.identical.to_dict(),
        )

    def test_44_shape_mismatch(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            compute_pentagon_scores(
                self.pose,
                self.pose[:29],
            )

    def test_45_wrong_joints(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            compute_pentagon_scores(
                np.zeros(
                    (30, 32, 3)
                ),
                self.pose,
            )

    def test_46_wrong_coords(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            compute_pentagon_scores(
                np.zeros(
                    (30, 33, 2)
                ),
                self.pose,
            )

    def test_47_nan(
        self,
    ):
        user = self.pose.copy()
        user[0, 0, 0] = np.nan

        with self.assertRaises(
            ValueError
        ):
            compute_pentagon_scores(
                user,
                self.pose,
            )

    def test_48_inf(
        self,
    ):
        user = self.pose.copy()
        user[0, 0, 0] = np.inf

        with self.assertRaises(
            ValueError
        ):
            compute_pentagon_scores(
                user,
                self.pose,
            )

    def test_49_feedback_keys(
        self,
    ):
        self.assertEqual(
            set(
                self.identical
                .feedback_summary
            ),
            {
                "timing",
                "accuracy",
                "detail",
                "balance",
                "rhythm",
            },
        )

    def test_50_timing_feedback(
        self,
    ):
        self.assertEqual(
            self.identical
            .feedback_summary[
                "timing"
            ]["lag_frames"],
            self.identical.lag_frames,
        )

    def test_51_rhythm_feedback_correlation(
        self,
    ):
        self.assertEqual(
            self.identical
            .feedback_summary[
                "rhythm"
            ]["motion_correlation"],
            self.identical
            .component_results[
                "rhythm"
            ]["diagnostics"][
                "motion_correlation"
            ],
        )

    def test_52_rhythm_feedback_mae(
        self,
    ):
        self.assertEqual(
            self.identical
            .feedback_summary[
                "rhythm"
            ]["normalized_motion_mae"],
            self.identical
            .component_results[
                "rhythm"
            ]["diagnostics"][
                "normalized_motion_mae"
            ],
        )

    def test_53_reliability_type(
        self,
    ):
        self.assertIsInstance(
            self.identical.reliability,
            PentagonReliability,
        )

    def test_54_scores_type(
        self,
    ):
        self.assertIsInstance(
            self.identical.scores,
            PentagonScores,
        )

    def test_55_weights_type(
        self,
    ):
        self.assertIsInstance(
            self.identical.weights,
            PentagonWeights,
        )

    def test_56_result_type(
        self,
    ):
        self.assertIsInstance(
            self.identical,
            PentagonScoreResult,
        )

    def test_57_overall_and(
        self,
    ):
        reliability = (
            self.identical.reliability
        )

        expected = all(
            (
                reliability.timing,
                reliability.accuracy,
                reliability.detail,
                reliability.balance,
                reliability.rhythm,
            )
        )

        self.assertEqual(
            reliability.overall,
            expected,
        )

    def stationary_result(
        self,
    ):
        pose = np.repeat(
            make_pose()[:1],
            30,
            axis=0,
        )

        return compute_pentagon_scores(
            pose,
            pose,
        )

    def test_58_unreliable_order(
        self,
    ):
        self.assertEqual(
            self.stationary_result()
            .reliability
            .unreliable_components,
            [
                "timing",
                "rhythm",
            ],
        )

    def test_59_stationary_rhythm_fallback(
        self,
    ):
        result = (
            self.stationary_result()
        )

        self.assertEqual(
            result.component_results[
                "rhythm"
            ]["diagnostics"][
                "fallback_reason"
            ],
            "both_constant",
        )

    def test_60_stationary_rhythm_unreliable(
        self,
    ):
        self.assertFalse(
            self.stationary_result()
            .reliability
            .rhythm
        )

    def test_61_stationary_overall_unreliable(
        self,
    ):
        self.assertFalse(
            self.stationary_result()
            .reliability
            .overall
        )

    def test_62_unreliable_score_still_used(
        self,
    ):
        self.assertEqual(
            self.stationary_result()
            .rule_score,
            100.0,
        )

    def test_63_custom_config_forwarded(
        self,
    ):
        config = replace(
            PentagonConfig(),
            use_z=False,
            timing_penalty_per_frame=7,
            rhythm_mae_scale=33,
        )

        result = compute_pentagon_scores(
            self.pose,
            self.pose,
            config,
        )

        self.assertFalse(
            result.component_results[
                "accuracy"
            ]["diagnostics"]["use_z"]
        )

        self.assertEqual(
            result.component_results[
                "timing"
            ]["diagnostics"][
                "penalty_per_frame"
            ],
            7,
        )

        self.assertEqual(
            result.component_results[
                "rhythm"
            ]["diagnostics"][
                "rhythm_mae_scale"
            ],
            33,
        )

    def test_64_json_basic_types(
        self,
    ):
        self.assertFalse(
            contains_numpy(
                self.identical.to_dict()
            )
        )

    def test_65_no_raw_pose(
        self,
    ):
        self.assertFalse(
            any(
                contains_key(
                    self.identical.to_dict(),
                    key,
                )
                for key in (
                    "user_seq",
                    "idol_seq",
                    "pose",
                )
            )
        )

    def test_66_weight_validation_names(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            _validate_weights(
                {"accuracy": 1.0}
            )

    def test_67_weight_validation_sum(
        self,
    ):
        bad_weights = dict(
            PENTAGON_SCORE_WEIGHTS
        )

        bad_weights["accuracy"] = 0.31

        with self.assertRaises(
            ValueError
        ):
            _validate_weights(
                bad_weights
            )

    def test_68_score_validation_range(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            _validate_score_value(
                101,
                "test",
            )

    def test_69_json_helper_numpy(
        self,
    ):
        result = _to_json_compatible(
            {
                "x": np.array(
                    [np.float32(1)]
                )
            }
        )

        self.assertEqual(
            result,
            {"x": [1.0]},
        )


class BehavioralIntegrationTests(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls):
        cls.idol = make_pose()

        cls.perfect = (
            compute_pentagon_scores(
                cls.idol,
                cls.idol,
            )
        )

        cls.delayed = (
            compute_pentagon_scores(
                delayed(cls.idol),
                cls.idol,
            )
        )

    def test_70_delay_lag(
        self,
    ):
        self.assertEqual(
            self.delayed.lag_frames,
            3,
        )

    def test_71_delay_frames(
        self,
    ):
        self.assertEqual(
            self.delayed
            .aligned_frame_count,
            27,
        )

    def test_72_delay_timing_lower(
        self,
    ):
        self.assertLess(
            self.delayed.scores.timing,
            100.0,
        )

    def test_73_delay_accuracy_not_double_penalized(
        self,
    ):
        self.assertAlmostEqual(
            self.delayed
            .scores
            .accuracy,
            100.0,
            places=5,
        )

    def test_74_delay_detail_not_double_penalized(
        self,
    ):
        self.assertAlmostEqual(
            self.delayed
            .scores
            .detail,
            100.0,
            places=5,
        )

    def test_75_delay_balance_not_double_penalized(
        self,
    ):
        self.assertAlmostEqual(
            self.delayed
            .scores
            .balance,
            100.0,
            places=5,
        )

    def test_76_delay_rhythm_not_double_penalized(
        self,
    ):
        self.assertAlmostEqual(
            self.delayed
            .scores
            .rhythm,
            100.0,
            places=5,
        )

    def test_77_delay_rule_formula(
        self,
    ):
        expected = sum(
            getattr(
                self.delayed.scores,
                name,
            )
            * PENTAGON_SCORE_WEIGHTS[
                name
            ]
            for name
            in PENTAGON_SCORE_NAMES
        )

        self.assertAlmostEqual(
            self.delayed.rule_score,
            expected,
        )

    def test_78_delay_only_timing_deduction(
        self,
    ):
        expected = (
            100.0
            - (
                100.0
                - self.delayed
                .scores
                .timing
            )
            * 0.15
        )

        self.assertAlmostEqual(
            self.delayed.rule_score,
            expected,
        )

    def test_79_wrist_error(
        self,
    ):
        user = self.idol.copy()
        user[:, 15, 0] += 0.2

        result = compute_pentagon_scores(
            user,
            self.idol,
        )

        self.assertLess(
            result.scores.accuracy,
            100.0,
        )

        self.assertLess(
            result.scores.detail,
            100.0,
        )

        self.assertLess(
            result.final_score,
            100.0,
        )

    def test_80_balance_error(
        self,
    ):
        user = self.idol.copy()
        user[:, 12, 1] += 0.25

        result = compute_pentagon_scores(
            user,
            self.idol,
        )

        self.assertLess(
            result.scores.balance,
            100.0,
        )

        self.assertLess(
            result.final_score,
            100.0,
        )

    def test_81_rhythm_error(
        self,
    ):
        user = make_pose(
            PATTERN[::-1]
        )

        result = compute_pentagon_scores(
            user,
            self.idol,
        )

        self.assertLess(
            result.scores.rhythm,
            100.0,
        )

        self.assertLess(
            result.final_score,
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
