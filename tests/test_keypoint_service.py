import numpy as np
import pytest

from app.services.keypoint_service import (
    KEYPOINT_DIM,
    NUM_LANDMARKS,
    compute_feedback,
    load_reference_keypoints,
)

FPS = 30.0


def _make_reference(num_frames: int) -> np.ndarray:
    """프레임마다 값이 다른 더미 reference (관절 좌표엔 관심 없고 frame_idx 계산만 검증)."""
    return np.zeros((num_frames, NUM_LANDMARKS, KEYPOINT_DIM), dtype=np.float32)


def _incoming() -> np.ndarray:
    return np.zeros((NUM_LANDMARKS, KEYPOINT_DIM), dtype=np.float32)


def test_frame_idx_wraps_instead_of_clamping_past_reference_length():
    """재생 시간이 기준 영상 길이를 넘어가면 마지막 프레임에 고정되지 않고 순환해야 한다.

    회귀 대상: keypoint_service.py의 timestamp_ms 보정이 원래 min()으로 클램프돼
    있어서, 세션이 기준 영상보다 길게 진행되면 frame_idx가 마지막 프레임에 영원히
    고정되는 버그가 있었다.
    """
    reference = _make_reference(num_frames=5)

    # frame 7 시점(ms) at 30fps → 5프레임짜리 기준 영상에서는 7 % 5 = 2번이어야 한다.
    timestamp_ms = int(7 * 1000 / FPS)
    result = compute_feedback(reference, _incoming(), frame_idx=0, timestamp_ms=timestamp_ms, fps=FPS)

    assert result["frame_idx"] == 2
    assert result["frame_idx"] != 4  # 클램프됐다면 마지막 인덱스(4)에 고정됐을 것


def test_frame_idx_keeps_wrapping_over_many_cycles():
    """기준 영상 길이를 여러 바퀴 넘게 진행해도 계속 순환해야 한다 (한 번만 순환하고
    다시 고정되는 회귀를 방지)."""
    reference = _make_reference(num_frames=5)

    for i in [5, 12, 23, 100]:  # 각각 5로 나눈 나머지: 0, 2, 3, 0
        timestamp_ms = int(i * 1000 / FPS)
        result = compute_feedback(
            reference, _incoming(), frame_idx=0, timestamp_ms=timestamp_ms, fps=FPS
        )
        assert result["frame_idx"] == i % 5


def test_frame_idx_has_no_rounding_jitter_at_steady_30fps():
    """클라이언트가 정확히 30fps로 균등하게 보낼 때, frame_idx가 반복되거나
    건너뛰지 않고 1씩 정확히 증가해야 한다.

    회귀 대상: timestamp_ms → frame_idx 변환에서 int() 절삭을 쓰면 왕복 오차가
    누적돼 300프레임 중 약 100번은 인덱스가 중복되고 약 99번은 하나씩 건너뛰었다
    (round()로 수정 전 실측). round()는 이 누적 오차를 없앤다.
    """
    reference = _make_reference(num_frames=1000)  # 순환 경계에 안 걸리도록 충분히 길게

    frame_indices = []
    for i in range(300):
        timestamp_ms = int(i * 1000 / FPS)
        result = compute_feedback(
            reference, _incoming(), frame_idx=0, timestamp_ms=timestamp_ms, fps=FPS
        )
        frame_indices.append(result["frame_idx"])

    diffs = [b - a for a, b in zip(frame_indices, frame_indices[1:])]
    assert all(d == 1 for d in diffs), f"프레임 인덱스가 1씩 증가하지 않음: {diffs[:20]}..."


@pytest.mark.asyncio
async def test_load_reference_keypoints_uses_an_existing_local_npy_before_redis(tmp_path):
    reference = _make_reference(num_frames=2)
    local_reference = tmp_path / "reference.npy"
    np.save(local_reference, reference)

    class RedisMustNotBeUsed:
        async def get(self, _key):
            raise AssertionError("local reference must be loaded before Redis")

    loaded = await load_reference_keypoints(RedisMustNotBeUsed(), 1, str(local_reference))

    assert np.array_equal(loaded, reference)
