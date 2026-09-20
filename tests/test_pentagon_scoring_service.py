from app.services.pentagon_scoring_service import (
    aggregate_pentagon_results,
    window_has_reference_seam,
)


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


def test_no_seam_for_single_frame_reference():
    """기준 영상이 1프레임이면 순환 개념 자체가 없으므로 이음매 아님."""
    assert not window_has_reference_seam([0] * 30, num_frames=1)


def test_aggregate_returns_none_when_all_windows_skipped():
    """모든 윈도우가 이음매 때문에 스킵돼 결과가 하나도 없으면 None을 반환해야 한다."""
    assert aggregate_pentagon_results([]) is None
