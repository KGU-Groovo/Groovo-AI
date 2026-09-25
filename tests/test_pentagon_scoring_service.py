import numpy as np
import pytest

from app.pentagon.pentagon_config import PentagonConfig
from app.services.pentagon_scoring_service import (
    WINDOW_SIZE,
    aggregate_pentagon_results,
    score_window,
    window_has_reference_seam,
)

NUM_LANDMARKS, KEYPOINT_DIM, WINDOW = 33, 3, 30


def _skeleton_sequence(seed: int = 0) -> np.ndarray:
    """관절마다 위치가 다른 사람 형태의 더미 30프레임 시퀀스 (어깨너비 등이
    전부 0이 되는 퇴화 케이스를 피하기 위해 관절 인덱스로 x좌표를 벌려준다)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, WINDOW)
    seq = np.zeros((WINDOW, NUM_LANDMARKS, KEYPOINT_DIM), dtype=np.float32)
    for j in range(NUM_LANDMARKS):
        phase = j * 0.15
        seq[:, j, 0] = j * 0.02 + 0.05 * np.sin(t + phase)
        seq[:, j, 1] = 0.5 + 0.05 * np.cos(t + phase)
        seq[:, j, 2] = 0.1 * np.sin(t * 0.5 + phase)
    return seq


def test_window_size_matches_pentagon_config_expected_frames():
    """websocket.py의 WINDOW_SIZE와 pentagon_scoring의 PentagonConfig.expected_frames가
    어긋나면(둘 중 하나만 리팩터링돼도) score_window가 매번 ValueError로 실패해서
    pentagon_scores가 항상 None이 되는 조용한 회귀가 생긴다. 두 상수가 항상 같은
    값을 쓰는지 여기서 고정해둔다."""
    assert WINDOW_SIZE == PentagonConfig().expected_frames


def test_no_seam_when_reference_index_increases_normally():
    """기준 영상이 윈도우보다 길어 순환 없이 쭉 증가하면 이음매가 아니다."""
    assert not window_has_reference_seam(list(range(0, 30)), num_frames=100)


def test_no_seam_with_occasional_repeat_from_fps_mismatch():
    """fps가 다르면 인덱스가 가끔 반복될 수 있는데, 이건 이음매가 아니다."""
    # 기준 fps가 사용자 전송 fps의 절반이라 인덱스가 한 프레임씩 걸러서 증가
    indices = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert not window_has_reference_seam(indices, num_frames=100)


def test_seam_detected_when_index_wraps_from_end_to_start():
    """기준 영상이 윈도우보다 짧아 마지막 프레임에서 처음으로 튀면 이음매로 감지."""
    # num_frames=20짜리 기준 영상이 윈도우 중간에 한 바퀴 순환
    indices = list(range(10, 20)) + list(range(0, 20))
    assert window_has_reference_seam(indices, num_frames=20)


def test_no_false_positive_seam_on_minor_reorder_in_long_reference():
    """기준 영상이 충분히 길면, 네트워크 재정렬로 인한 사소한 순서 뒤바뀜이나
    미세한 지터는 이음매로 오탐하면 안 된다."""
    jitter = [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10]
    assert not window_has_reference_seam(jitter, num_frames=200)

    minor_swap = [0, 1, 2, 3, 5, 4, 6, 8, 7, 9, 10, 12, 11, 13, 14, 15]
    assert not window_has_reference_seam(minor_swap, num_frames=200)


def test_no_false_positive_seam_on_backward_seek_in_long_reference():
    """기준 영상이 충분히 길면, 실제 순환이 아닌 되감기(뒤로 점프)라도 그 폭이
    기준 영상 길이의 절반을 안 넘으면 이음매로 오탐하면 안 된다."""
    seek_back = list(range(0, 20)) + list(range(0, 10))
    assert not window_has_reference_seam(seek_back, num_frames=200)


def test_no_seam_for_single_frame_reference():
    """기준 영상이 1프레임이면 순환 개념 자체가 없으므로 이음매 아님."""
    assert not window_has_reference_seam([0] * 30, num_frames=1)


def test_aggregate_returns_none_when_all_windows_skipped():
    """모든 윈도우가 이음매 때문에 스킵돼 결과가 하나도 없으면 None을 반환해야 한다."""
    assert aggregate_pentagon_results([]) is None


def test_aggregate_computes_exact_arithmetic_mean_across_windows():
    """윈도우별 결과를 단순 산술 평균으로 집계하는지 정확한 숫자로 검증한다
    (가중 평균이나 다른 집계 방식으로 잘못 바뀌는 회귀를 방지)."""
    window_results = [
        {"final_score": 100.0, "scores": {"accuracy": 100.0, "detail": 90.0, "balance": 80.0, "timing": 70.0, "rhythm": 60.0}},
        {"final_score": 50.0, "scores": {"accuracy": 50.0, "detail": 50.0, "balance": 50.0, "timing": 50.0, "rhythm": 50.0}},
        {"final_score": 0.0, "scores": {"accuracy": 0.0, "detail": 10.0, "balance": 20.0, "timing": 30.0, "rhythm": 40.0}},
    ]

    result = aggregate_pentagon_results(window_results)

    assert result["window_count"] == 3
    assert result["final_score"] == 50.0  # (100+50+0)/3
    assert result["scores"] == {
        "accuracy": 50.0,   # (100+50+0)/3
        "detail": 50.0,     # (90+50+10)/3
        "balance": 50.0,    # (80+50+20)/3
        "timing": 50.0,     # (70+50+30)/3
        "rhythm": 50.0,     # (60+50+40)/3
    }


def test_score_window_raises_on_untracked_person():
    """MediaPipe가 사람을 못 잡아 전부 [0,0,0]으로 오면 pentagon_scoring이 명확한
    ValueError를 던져야 한다 (호출부 websocket.py가 이걸 잡아서 윈도우만 스킵함).
    """
    zeros = [np.zeros((NUM_LANDMARKS, KEYPOINT_DIM), dtype=np.float32) for _ in range(WINDOW)]
    ref = list(_skeleton_sequence())
    with pytest.raises(ValueError):
        score_window(zeros, ref)


def test_score_window_raises_on_nan_input():
    """좌표에 NaN이 섞여도(추적 실패 등) 조용히 잘못된 점수를 내지 않고 예외를
    던져야 한다."""
    user = _skeleton_sequence(seed=1)
    user[0, 0, 0] = np.nan
    ref = list(_skeleton_sequence())
    with pytest.raises(ValueError):
        score_window(list(user), ref)


def test_score_window_survives_realistic_noise_without_crashing():
    """실제 웹캠 캡처처럼 좌표에 소량의 가우시안 노이즈가 껴도 크래시 없이
    0~100 범위의 유효한 점수를 내야 한다."""
    rng = np.random.default_rng(7)
    ref = _skeleton_sequence()

    for noise_scale in (0.005, 0.02, 0.05):
        noisy_user = ref + rng.normal(0, noise_scale, ref.shape).astype(np.float32)
        result = score_window(list(noisy_user), list(ref))
        assert 0 <= result["final_score"] <= 100
        for score in result["scores"].values():
            assert 0 <= score <= 100


def test_score_window_corrects_different_camera_aspect_ratios():
    reference = _skeleton_sequence()
    hip_center = 0.5 * (reference[:, 23:24] + reference[:, 24:25])
    user = hip_center + (reference - hip_center) * np.array([0.82, 1.0, 1.0], dtype=np.float32)

    result = score_window(
        list(user),
        list(reference),
        user_aspect_ratio=1 / 0.82,
        reference_aspect_ratio=1.0,
    )

    assert result["scores"]["accuracy"] > 99.0
