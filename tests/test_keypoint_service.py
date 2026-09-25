import numpy as np
import pytest

from app.config import settings
from app.services.keypoint_service import (
    KEYPOINT_DIM,
    NUM_LANDMARKS,
    compute_feedback,
    load_reference_keypoints,
)

FPS = 30.0

# MediaPipe Pose 인덱스 (사람이 서 있는 정면 자세를 대략 근사한 좌표).
_NOSE, _L_SHOULDER, _R_SHOULDER = 0, 11, 12
_L_ELBOW, _R_ELBOW, _L_WRIST, _R_WRIST = 13, 14, 15, 16
_L_HIP, _R_HIP = 23, 24
_L_KNEE, _R_KNEE, _L_ANKLE, _R_ANKLE = 25, 26, 27, 28

_STANDING_POSE = {
    _NOSE: (0.50, 0.15, 0.0),
    _L_SHOULDER: (0.42, 0.30, 0.0),
    _R_SHOULDER: (0.58, 0.30, 0.0),
    _L_ELBOW: (0.38, 0.45, 0.0),
    _R_ELBOW: (0.62, 0.45, 0.0),
    _L_WRIST: (0.35, 0.60, 0.0),
    _R_WRIST: (0.65, 0.60, 0.0),
    _L_HIP: (0.45, 0.55, 0.0),
    _R_HIP: (0.55, 0.55, 0.0),
    _L_KNEE: (0.44, 0.75, 0.0),
    _R_KNEE: (0.56, 0.75, 0.0),
    _L_ANKLE: (0.43, 0.95, 0.0),
    _R_ANKLE: (0.57, 0.95, 0.0),
}


def _pose(overrides: dict[int, tuple[float, float, float]] | None = None) -> np.ndarray:
    """실제 사람 골격에 가까운 정지 자세 한 프레임을 만든다.

    지정하지 않은 관절(얼굴 세부, 손가락 등)은 몸통 근처의 서로 다른 값으로
    채워서, 우연히 (0,0,0)이 되어 "추적 안 됨"으로 오인되지 않게 한다.
    """
    joints = np.zeros((NUM_LANDMARKS, KEYPOINT_DIM), dtype=np.float32)
    for idx in range(NUM_LANDMARKS):
        if idx in _STANDING_POSE:
            joints[idx] = _STANDING_POSE[idx]
        else:
            joints[idx] = (0.5 + 0.001 * idx, 0.5 + 0.001 * idx, 0.001 * idx)
    for idx, xyz in (overrides or {}).items():
        joints[idx] = xyz
    return joints


def _reference_of(pose: np.ndarray, num_frames: int = 10) -> np.ndarray:
    """모든 프레임이 동일한 정지 자세인 reference (프레임 인덱스는 무관, 채점 공식만 검증)."""
    return np.stack([pose] * num_frames).astype(np.float32)


# _STANDING_POSE는 좌우 대칭이라 왼쪽/오른쪽을 바꿔치기해도 똑같은 좌표가 나온다
# (미러링 오류를 탐지 못 해도 우연히 통과할 수 있음). 좌우 스왑 탐지를 실제로
# 검증하려면 좌우가 다른 비대칭 자세가 필요하다 — 오른팔만 위로 든 자세.
_ASYM_POSE_OVERRIDES = {
    _R_ELBOW: (0.62, 0.15, 0.0),
    _R_WRIST: (0.65, 0.05, 0.0),
}
_MIRRORED_ASYM_OVERRIDES = {
    _L_ELBOW: (0.38, 0.15, 0.0),
    _L_WRIST: (0.35, 0.05, 0.0),
    _R_ELBOW: (0.62, 0.45, 0.0),
    _R_WRIST: (0.65, 0.60, 0.0),
}


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


def test_feedback_aligns_to_the_nearest_matching_reference_frame():
    reference = np.zeros((9, NUM_LANDMARKS, KEYPOINT_DIM), dtype=np.float32)
    reference[3].fill(1.0)
    reference[5].fill(-1.0)

    # 재생 시점은 5번 프레임이지만, 사용자는 약 2프레임 늦은 3번 포즈를 하고 있다.
    timestamp_ms = round(5 * 1000 / FPS)
    result = compute_feedback(
        reference,
        reference[3],
        frame_idx=0,
        timestamp_ms=timestamp_ms,
        fps=FPS,
    )

    assert result["frame_idx"] == 3
    assert result["score"] > 0.99


def test_feedback_ignores_camera_translation_and_body_scale():
    pose = np.zeros((NUM_LANDMARKS, KEYPOINT_DIM), dtype=np.float32)
    pose[11, 0] = -1.0
    pose[12, 0] = 1.0
    pose[23, 0] = -0.5
    pose[24, 0] = 0.5
    pose[15, 1] = 0.7
    reference = np.repeat(pose[None, ...], 9, axis=0)

    # 같은 포즈를 더 멀리서 찍고 카메라 위치가 달라도 만점에 가까워야 한다.
    incoming = pose * 1.8 + np.array([4.0, -3.0, 0.5], dtype=np.float32)
    result = compute_feedback(reference, incoming, frame_idx=0)

    assert result["score"] > 0.99


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


# ── 아래부터는 채점 공식(코사인 유사도 + 골반너비 정규화 거리 감점) 자체를
# 검증하는 회귀 테스트. 2026-09-23 세션에서 실제 영상으로 발견한 버그들의
# 재발 방지용이며, 2026-09-24에 새 시나리오(골반 관절 자체가 추적 안 되는
# 경우)를 추가하다가 새 버그를 하나 더 발견해 같이 고쳤다.


def test_identical_pose_scores_perfect():
    pose = _pose()
    result = compute_feedback(_reference_of(pose), pose, frame_idx=0)
    assert result["score"] == 1.0
    assert result["message"] == "Good!"


def test_single_arm_offset_is_no_longer_masked_as_good():
    """회귀 대상: 관절 하나 합산 코사인 유사도로는 팔 하나만 계속 잘못 벌려도
    score 0.93~0.99, "Good!"이 나왔다(전체 33관절 평균에 희석). 관절별 가중
    코사인 + 최대오차 관절 감점 도입 후에는 good 임계값(0.80) 밑으로 떨어져야
    한다."""
    ref = _reference_of(_pose())
    # 오른쪽 팔꿈치/손목을 머리 위로 크게 들어올린 동작 오류.
    wrong_arm = _pose({_R_ELBOW: (0.62, 0.15, 0.0), _R_WRIST: (0.65, 0.05, 0.0)})

    result = compute_feedback(ref, wrong_arm, frame_idx=0)

    assert result["score"] < settings.feedback_threshold_good
    assert _R_WRIST in result["worst_joints"] or _R_ELBOW in result["worst_joints"]


def test_both_arms_wrong_scores_worse_than_one_arm_wrong():
    """오류 관절이 늘어나면 점수가 더 나빠져야 한다(오류 크기에 비례)."""
    ref = _reference_of(_pose())
    one_arm_wrong = _pose({_R_ELBOW: (0.62, 0.15, 0.0), _R_WRIST: (0.65, 0.05, 0.0)})
    both_arms_wrong = _pose(
        {
            _L_ELBOW: (0.38, 0.15, 0.0),
            _L_WRIST: (0.35, 0.05, 0.0),
            _R_ELBOW: (0.62, 0.15, 0.0),
            _R_WRIST: (0.65, 0.05, 0.0),
        }
    )

    score_one = compute_feedback(ref, one_arm_wrong, frame_idx=0)["score"]
    score_both = compute_feedback(ref, both_arms_wrong, frame_idx=0)["score"]

    assert score_both < score_one


def test_camera_distance_does_not_cause_false_negative():
    """회귀 대상: 정규화 없이는 사용자가 카메라에 더 가까이 서기만 해도(자세는
    동일) 거리 오차 감점이 커져서 오탐이 났다(실측: 1.3배 확대 시 score 평균
    0.83). 골반너비로 정규화한 뒤에는 스케일이 달라도 동일 자세면 만점이어야
    한다."""
    ref = _reference_of(_pose())
    closer_to_camera = _pose() * 1.3  # 몸 전체가 화면에 더 크게 잡힘, 자세는 동일

    result = compute_feedback(ref, closer_to_camera, frame_idx=0)

    assert result["score"] >= settings.feedback_threshold_good
    assert result["message"] == "Good!"


def test_screen_position_translation_does_not_affect_score():
    """골반 중점 기준 상대좌표라 화면상 위치(좌우/상하로 서 있는 위치)는
    점수에 영향을 주면 안 된다."""
    ref = _reference_of(_pose())
    shifted = _pose() + np.array([0.2, 0.1, 0.0], dtype=np.float32)

    result = compute_feedback(ref, shifted, frame_idx=0)

    assert result["score"] == 1.0


def test_single_untracked_joint_is_excluded_not_penalized():
    """MediaPipe가 관절 하나를 못 잡아 (0,0,0)으로 채운 경우, 나머지 자세가
    완벽히 일치하면 그 관절만 채점에서 빠지고 만점이 나와야 한다(추적 실패를
    "그 관절이 원점으로 순간이동한 큰 오차"로 오인하면 안 됨)."""
    ref = _reference_of(_pose())
    incoming = _pose()
    incoming[_L_KNEE] = (0.0, 0.0, 0.0)

    result = compute_feedback(ref, incoming, frame_idx=0)

    assert result["score"] == 1.0
    assert _L_KNEE not in result["worst_joints"]


def test_untracked_hip_does_not_corrupt_whole_frame_scoring():
    """2026-09-24에 새로 찾은 버그의 회귀 테스트.

    골반은 좌표계의 원점(_center_on_hip)을 정하는 기준이라, 골반 관절 하나가
    추적 실패((0,0,0))하면 기존 코드는 그대로 (실제 골반 + (0,0,0)) / 2로
    중점을 계산해 원점 자체가 이미지 좌상단 쪽으로 끌려갔다. 그 결과 나머지
    32개 관절은 실제로 완벽히 일치하는데도 score가 1.0 → -0.20까지 떨어졌다
    (관절별 untracked 가드는 "그 관절의 가중치"만 0으로 만들 뿐, 원점 계산
    오염 자체는 못 막았기 때문). 반대쪽 골반을 원점으로 쓰고, reference도
    같은 기준점으로 맞춰 비교하도록 고쳤다.
    """
    ref = _reference_of(_pose())
    incoming = _pose()
    incoming[_L_HIP] = (0.0, 0.0, 0.0)

    result = compute_feedback(ref, incoming, frame_idx=0)

    assert result["score"] >= settings.feedback_threshold_good
    assert result["message"] == "Good!"
    assert _L_HIP not in result["worst_joints"]


def test_both_hips_untracked_does_not_crash_or_produce_nan():
    """양쪽 골반이 모두 추적 실패한 극단적 경우(원점을 구할 방법이 없음)에도
    예외나 NaN 없이 진행되어야 한다 — 실시간 세션이 이 프레임 하나 때문에
    끊기면 안 된다."""
    ref = _reference_of(_pose())
    incoming = _pose()
    incoming[_L_HIP] = (0.0, 0.0, 0.0)
    incoming[_R_HIP] = (0.0, 0.0, 0.0)

    result = compute_feedback(ref, incoming, frame_idx=0)

    assert -1.0 <= result["score"] <= 1.0
    assert not np.isnan(result["score"])


def test_degenerate_hip_width_skips_distance_penalty_without_crashing():
    """몸을 거의 옆으로 돌려 골반 두 점이 화면상 거의 겹치는 경우(카메라 거리
    추정 불가) 거리 기반 감점은 스킵하고 코사인만으로 채점해야 하며, 0에
    가까운 값으로 나누다 발산하거나 예외가 나면 안 된다."""
    ref = _reference_of(_pose())
    incoming = _pose({_L_HIP: (0.499, 0.55, 0.0), _R_HIP: (0.501, 0.55, 0.0)})

    result = compute_feedback(ref, incoming, frame_idx=0)

    assert -1.0 <= result["score"] <= 1.0
    assert not np.isnan(result["score"])


def test_fully_untracked_person_does_not_crash_or_produce_nan():
    """사람 전체가 화면 밖으로 나가 33관절이 모두 (0,0,0)인 극단적 프레임도
    예외 없이 처리되어야 한다."""
    ref = _reference_of(_pose())
    incoming = np.zeros((NUM_LANDMARKS, KEYPOINT_DIM), dtype=np.float32)

    result = compute_feedback(ref, incoming, frame_idx=0)

    assert -1.0 <= result["score"] <= 1.0
    assert not np.isnan(result["score"])


def test_single_leg_joint_offset_is_penalized_like_arm_offset():
    """다리 쪽 오류도 팔과 마찬가지로 good 임계값 밑으로 떨어져야 한다(관절
    가중치가 팔/다리 양쪽에 동일하게 적용되는지 확인)."""
    ref = _reference_of(_pose())
    wrong_ankle = _pose({_R_ANKLE: (0.57, 0.70, 0.0)})

    result = compute_feedback(ref, wrong_ankle, frame_idx=0)

    assert result["score"] < settings.feedback_threshold_good
    assert _R_ANKLE in result["worst_joints"]


def test_natural_tracking_noise_still_scores_good():
    """MediaPipe 트래킹 자체의 미세한 프레임 간 흔들림(노이즈) 수준에서는
    동작이 실제로 같으면 여전히 "Good!"이 나와야 한다 — 감점 항이 너무
    민감해서 정상 동작까지 오탐하면 안 된다."""
    ref = _reference_of(_pose())
    rng = np.random.default_rng(0)
    noisy = _pose() + rng.normal(0, 0.005, size=(NUM_LANDMARKS, KEYPOINT_DIM)).astype(np.float32)

    result = compute_feedback(ref, noisy, frame_idx=0)

    assert result["score"] >= settings.feedback_threshold_good
    assert result["message"] == "Good!"


def test_score_decreases_monotonically_as_offset_grows():
    """오류 크기가 커질수록 점수가 계속 나빠져야 한다(코사인+거리 감점을
    합친 공식이 중간 어딘가에서 반등하는 비정상적인 비단조 구간이 없는지
    확인)."""
    ref = _reference_of(_pose())
    scores = []
    for magnitude in (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.5):
        offset_pose = _pose({_R_WRIST: (0.65, 0.60 - magnitude, 0.0)})
        scores.append(compute_feedback(ref, offset_pose, frame_idx=0)["score"])

    assert scores == sorted(scores, reverse=True), f"비단조 구간 발견: {scores}"


def test_extreme_offset_clips_to_negative_one_without_crashing():
    """말도 안 되게 큰 오차(관절이 화면 밖 멀리 순간이동)에서도 score가
    -1.0 아래로 내려가지 않고 클리핑되어야 한다."""
    ref = _reference_of(_pose())
    teleported = _pose({_R_WRIST: (5.0, -5.0, 3.0)})

    result = compute_feedback(ref, teleported, frame_idx=0)

    assert result["score"] == -1.0


def test_untracked_high_weight_joint_is_excluded_without_corrupting_scale():
    """손목/발목처럼 가중치가 높은 관절이 추적 실패해도(골반과 달리 원점
    계산에는 안 쓰이므로) 그 관절만 빠지고 나머지는 정상적으로 만점이어야
    한다."""
    ref = _reference_of(_pose())
    incoming = _pose()
    incoming[_R_WRIST] = (0.0, 0.0, 0.0)

    result = compute_feedback(ref, incoming, frame_idx=0)

    assert result["score"] == 1.0
    assert _R_WRIST not in result["worst_joints"]


def test_untracked_hip_does_not_mask_a_real_error_in_the_same_frame():
    """2026-09-24 시나리오 확장 중 새로 찾은 두 번째 버그의 회귀 테스트.

    골반 하나가 추적 실패한 프레임에서 실제 팔 오류가 동시에 났을 때, 처음
    고친 버전은 "골반 추적 실패 = 몸 크기 신뢰 불가"로 보고 거리 기반 감점을
    통째로 꺼버렸다. 그 결과 팔이 완전히 다른 방향을 가리키는 명백한 오류가
    score 0.37("다시 시도해 보세요")에서 0.88("Good!")로 올라가며 그대로
    통과해버렸다 — 골반 추적 글리치가 실제 오류를 가리는 새로운 오탐 경로였다.
    골반 한쪽만 없고 reference는 정상인 흔한 경우엔, "골반너비 : 어깨너비"
    비율(reference 기준)로 incoming의 골반너비를 추정해 감점을 계속 적용하도록
    고쳤다. 이 테스트는 골반 추적 실패 여부와 무관하게 점수가 크게 갈리지
    않아야 함을 확인한다.
    """
    ref = _reference_of(_pose())
    arm_error = {_R_ELBOW: (0.62, 0.15, 0.0), _R_WRIST: (0.65, 0.05, 0.0)}

    hips_tracked = _pose(arm_error)
    score_hips_tracked = compute_feedback(ref, hips_tracked, frame_idx=0)["score"]

    hip_glitch_same_error = _pose(arm_error)
    hip_glitch_same_error[_L_HIP] = (0.0, 0.0, 0.0)
    result_with_glitch = compute_feedback(ref, hip_glitch_same_error, frame_idx=0)

    assert result_with_glitch["score"] < settings.feedback_threshold_bad
    assert abs(result_with_glitch["score"] - score_hips_tracked) < 0.1


def test_worst_joints_ranks_largest_error_first():
    """크기가 다른 오류 세 개를 동시에 주면, worst_joints 1순위는 실제로
    가장 많이 틀린 관절이어야 한다."""
    ref = _reference_of(_pose())
    incoming = _pose(
        {
            _L_ANKLE: (0.43, 0.85, 0.0),  # 작은 오류
            _R_ELBOW: (0.62, 0.30, 0.0),  # 중간 오류
            _R_WRIST: (0.65, 0.10, 0.0),  # 큰 오류
        }
    )

    result = compute_feedback(ref, incoming, frame_idx=0)

    assert result["worst_joints"][0] == _R_WRIST


def test_reference_hip_untracked_defensive_fallback_does_not_crash():
    """reference는 사전 처리된 curated 데이터라 골반이 빠질 일이 거의 없지만,
    방어적으로 넣은 처리 경로도 크래시나 NaN 없이 동작해야 한다. 이 경우는
    어깨너비 비율을 역산할 기준(정상 reference)이 없으므로 거리 감점은
    스킵되고 코사인만으로 채점된다."""
    bad_ref_frame = _pose()
    bad_ref_frame[_L_HIP] = (0.0, 0.0, 0.0)
    ref = _reference_of(bad_ref_frame)

    result = compute_feedback(ref, _pose(), frame_idx=0)

    assert -1.0 <= result["score"] <= 1.0
    assert not np.isnan(result["score"])


def test_left_right_mirrored_pose_is_not_scored_as_correct():
    """좌우가 완전히 뒤바뀐 동작(오른팔을 들어야 하는데 왼팔을 든 경우)은
    실제로 틀린 동작이므로 good 임계값 밑으로 떨어져야 한다. 대칭인
    _STANDING_POSE만으로는 좌우를 바꿔도 우연히 같은 좌표가 나와 이 종류의
    오류를 검증할 수 없어서, 오른팔만 든 비대칭 자세로 확인한다."""
    ref = _reference_of(_pose(_ASYM_POSE_OVERRIDES))
    mirrored = _pose(_MIRRORED_ASYM_OVERRIDES)

    result = compute_feedback(ref, mirrored, frame_idx=0)

    assert result["score"] < settings.feedback_threshold_good


def test_both_hips_untracked_falls_back_to_shoulder_anchor_not_absolute_coords():
    """2026-09-24 시나리오 확장 중 새로 찾은 세 번째 버그(가장 심각함)의
    회귀 테스트.

    골반이 양쪽 다 추적 실패하면 기존 코드는 원점을 (0,0,0)으로 둬서,
    사실상 이 채점 공식을 새로 만들게 된 최초의 버그(_center_on_hip 주석
    참고 — 화면 절대좌표로 코사인 유사도를 재면 자세가 달라도 거의 안
    변함)로 조용히 되돌아갔다. 실측: 좌우가 완전히 뒤바뀐 틀린 동작이
    골반이 둘 다 안 잡히면 score가 0.29("다시 시도") → 0.95("Good!")로
    올라가며 명백한 오류를 완전히 놓쳤다. 골반이 둘 다 없으면 어깨(11/12)
    중점을 대신 원점으로 쓰도록 고쳐서, 적어도 "화면 절대좌표로 후퇴"하는
    최악의 경우는 막았다(완전히 골반 있을 때와 똑같은 민감도까지는 아니지만
    — 거리 기반 감점은 이 경우 스킵됨 — 최소한 명백한 오류를 "Good!"으로
    잘못 통과시키지는 않는다)."""
    ref = _reference_of(_pose(_ASYM_POSE_OVERRIDES))
    mirrored_with_hip_glitch = _pose(_MIRRORED_ASYM_OVERRIDES)
    mirrored_with_hip_glitch[_L_HIP] = (0.0, 0.0, 0.0)
    mirrored_with_hip_glitch[_R_HIP] = (0.0, 0.0, 0.0)

    result = compute_feedback(ref, mirrored_with_hip_glitch, frame_idx=0)

    assert result["score"] < settings.feedback_threshold_good
    assert result["message"] != "Good!"


def test_both_hips_untracked_with_correct_pose_still_scores_perfect():
    """골반이 둘 다 안 잡혀도(어깨 대체 원점 사용) 실제 자세가 완벽히
    일치하면 여전히 만점이 나와야 한다 — 어깨 대체가 불필요한 감점을
    만들면 안 된다."""
    ref = _reference_of(_pose())
    incoming = _pose()
    incoming[_L_HIP] = (0.0, 0.0, 0.0)
    incoming[_R_HIP] = (0.0, 0.0, 0.0)

    result = compute_feedback(ref, incoming, frame_idx=0)

    assert result["score"] == 1.0
    assert result["message"] == "Good!"


def test_all_high_weight_joints_untracked_simultaneously_does_not_crash():
    """손목/발목 4개(가중치 3.0)가 동시에 전부 추적 실패해도 — 가중치 정규화
    분모(joint_weights.max())가 급격히 낮은 값(1.0)으로 바뀌는 경계 상황 —
    크래시나 이상값 없이, 남은 관절이 일치하면 만점이 나와야 한다."""
    ref = _reference_of(_pose())
    incoming = _pose()
    for joint in (_L_WRIST, _R_WRIST, _L_ANKLE, _R_ANKLE):
        incoming[joint] = (0.0, 0.0, 0.0)

    result = compute_feedback(ref, incoming, frame_idx=0)

    assert result["score"] == 1.0
    assert not np.isnan(result["score"])


def test_depth_only_shift_does_not_affect_score():
    """카메라와의 앞뒤 거리(z축 전체 이동)만 달라지고 실제 자세(x,y와 상대적
    깊이)는 동일하면 골반 중심화로 상쇄되어 만점이어야 한다."""
    ref = _reference_of(_pose())
    incoming = _pose().copy()
    incoming[:, 2] += 0.3

    result = compute_feedback(ref, incoming, frame_idx=0)

    assert result["score"] == 1.0


def test_both_hips_untracked_with_real_error_is_caught_by_shoulder_width_penalty():
    """2026-09-24 실제 영상(395프레임 댄스 챌린지 클립)으로 both-hips-untracked
    시나리오를 검증하다가, 어깨를 원점으로만 대체하고 거리 감점을 계속
    스킵하면 여전히 부족하다는 걸 발견했다 — 골반이 둘 다 없고 실제 팔 오류가
    겹치면 395프레임 내내 "Good!"으로 통과했다(관절 가중 코사인만으로는
    부족하다는 게 애초에 거리 감점을 도입한 이유였는데, 골반이 없다고 감점을
    끄면 결국 그 한계로 되돌아감). 골반이 둘 다 없어도 어깨너비 자체는 구할 수
    있으므로, 이 경우엔 골반너비로 환산하지 않고 어깨너비를 그대로 거리 감점의
    스케일 기준으로 쓰도록 보강했다."""
    ref = _reference_of(_pose(_ASYM_POSE_OVERRIDES))
    mirrored_with_no_hips = _pose(_MIRRORED_ASYM_OVERRIDES)
    mirrored_with_no_hips[_L_HIP] = (0.0, 0.0, 0.0)
    mirrored_with_no_hips[_R_HIP] = (0.0, 0.0, 0.0)

    result = compute_feedback(ref, mirrored_with_no_hips, frame_idx=0)

    assert result["score"] < settings.feedback_threshold_bad
    assert result["message"] != "Good!"


def test_hips_and_shoulders_untracked_falls_back_to_ear_anchor_and_still_catches_errors():
    """2026-09-24: 골반과 어깨가 동시에 전부 추적 실패하는 경우, 처음엔
    "대체할 몸통 기준점이 없다"며 알려진 한계로 남겨뒀다(원점이 (0,0,0)으로
    후퇴해 화면 절대좌표 코사인이 되므로, 실제 영상으로 재현해보니 손목 고정
    오류가 395프레임 내내 "Good!"으로 통과했었다). 사용자 요청으로 다시 파본
    결과, MediaPipe 33관절엔 귀(7/8)·눈 바깥쪽(3/6)처럼 골반/어깨와 마찬가지로
    좌우 대칭인 얼굴 랜드마크가 더 있다는 걸 활용해 원점 우선순위를 골반→어깨→
    귀→눈으로 확장했고, 거리 감점의 스케일도 같은 방식(귀/눈 사이 거리를
    그대로 사용)으로 확장했다. 실제 영상 재검증 결과 같은 조합(골반+어깨 미검출
    + 손목 고정 오류)의 good 비율이 100% → 20%로 개선됐다(정상 트래킹 기준선
    31.6%보다 오히려 더 예민한 정도 — 관대한 쪽이 아니라 안전한 쪽으로 치우침).
    """
    ref = _reference_of(_pose(_ASYM_POSE_OVERRIDES))
    mirrored_no_torso = _pose(_MIRRORED_ASYM_OVERRIDES)
    for joint in (_L_HIP, _R_HIP, _L_SHOULDER, _R_SHOULDER):
        mirrored_no_torso[joint] = (0.0, 0.0, 0.0)

    result = compute_feedback(ref, mirrored_no_torso, frame_idx=0)

    assert result["score"] < settings.feedback_threshold_good
    assert result["message"] != "Good!"


def test_no_bilateral_landmark_at_all_does_not_crash_true_known_limitation():
    """골반·어깨·귀·눈(바깥쪽)까지 우선순위 목록의 모든 좌우 쌍이 전부
    추적 실패하면, 정말로 원점을 구할 방법이 없어 (0,0,0)(=화면 절대좌표
    코사인)으로 후퇴한다 — 이 경우엔 실제 오류도 놓칠 수 있는 진짜 한계다.
    다만 실제 MediaPipe 출력에서 얼굴·몸통 랜드마크가 전부 안 잡히는데 손목
    같은 말단 관절만 계속 잡히는 조합은 현실적으로 거의 없다. 이 테스트는
    그 한계를 문서화하고, 적어도 크래시나 NaN은 없음을 보장한다."""
    ref = _reference_of(_pose(_ASYM_POSE_OVERRIDES))
    incoming = _pose(_MIRRORED_ASYM_OVERRIDES)
    for joint in (_L_HIP, _R_HIP, _L_SHOULDER, _R_SHOULDER, 7, 8, 3, 6):
        incoming[joint] = (0.0, 0.0, 0.0)

    result = compute_feedback(ref, incoming, frame_idx=0)

    assert -1.0 <= result["score"] <= 1.0
    assert not np.isnan(result["score"])
