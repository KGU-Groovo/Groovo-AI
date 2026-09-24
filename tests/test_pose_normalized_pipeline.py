import numpy as np
import torch

from generate_data import compute_score_from_error, generate_synthetic_dataset_from_windows
from inference import run_single_inference
from normalization import normalize_pose_sequence


def _pose_window():
    sequence = np.zeros((30, 33, 3), dtype=np.float32)
    sequence[:, 11, 0] = -1.0
    sequence[:, 12, 0] = 1.0
    sequence[:, 23, 0] = -0.4
    sequence[:, 24, 0] = 0.4
    sequence[:, 15, 1] = np.linspace(0.0, 0.5, 30)
    return sequence


def test_synthetic_dataset_stores_pose_normalized_sequences():
    dataset = generate_synthetic_dataset_from_windows(
        _pose_window()[None, ...], variants_per_window=1, seed=7
    )

    shoulder_width = np.linalg.norm(
        dataset["idol_seqs"][0, :, 11, :] - dataset["idol_seqs"][0, :, 12, :], axis=1
    )

    assert np.allclose(shoulder_width, 1.0, atol=1e-6)


def test_sequence_normalization_uses_a_stable_scale_when_one_frame_has_zero_shoulder_width():
    sequence = _pose_window()
    sequence[10, 11] = sequence[10, 12]
    sequence[10, 15, 1] = 0.5

    normalized = normalize_pose_sequence(sequence)

    assert np.isfinite(normalized).all()
    assert np.abs(normalized).max() < 10.0


def test_normalized_error_score_keeps_small_errors_in_a_usable_score_range():
    small_error_score = compute_score_from_error(
        np.full(33, 0.65, dtype=np.float32), "noise_small"
    )
    large_error_score = compute_score_from_error(
        np.full(33, 1.55, dtype=np.float32), "noise_large"
    )

    assert 65.0 <= small_error_score <= 80.0
    assert small_error_score > large_error_score


def test_inference_normalizes_translation_before_model_prediction():
    reference = _pose_window()
    translated_user = reference + np.array([4.0, -3.0, 0.5], dtype=np.float32)

    class CaptureModel:
        def eval(self):
            return self

        def __call__(self, idol_seq, user_seq):
            self.idol_seq = idol_seq.detach().cpu().numpy()
            self.user_seq = user_seq.detach().cpu().numpy()
            return {
                "score_100": torch.tensor([90.0]),
                "score_norm": torch.tensor([0.9]),
                "highlight_joints": torch.tensor([[11, 13, 15]]),
                "pred_joint_errors": torch.zeros((1, 33)),
            }

    model = CaptureModel()
    run_single_inference(
        model, reference, translated_user, torch.device("cpu"), normalize_input=True
    )

    assert np.allclose(model.idol_seq, model.user_seq, atol=1e-6)
