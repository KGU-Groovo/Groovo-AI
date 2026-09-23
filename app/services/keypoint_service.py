import io
import json
import logging
from typing import Any

import aioboto3
import numpy as np
import redis.asyncio as aioredis
from botocore.config import Config as BotoConfig

from app.config import settings

logger = logging.getLogger(__name__)

# MediaPipe Pose 기준: 33 landmarks × [x, y, z]
NUM_LANDMARKS = 33
KEYPOINT_DIM = 3

# MediaPipe Pose 왼쪽/오른쪽 골반 인덱스 — 자세 비교의 원점으로 사용
_L_HIP = 23
_R_HIP = 24
# 2026-09-23: 몸 크기(카메라와의 거리) 정규화 기준으로 처음엔 pentagon_common.py와
# 같은 어깨너비(11/12)를 썼는데, 실제 영상으로 검증하다가 문제를 발견했다 —
# 어깨너비는 팔 동작 자체(어깨를 돌리거나 팔을 뻗는 동작)에도 크게 흔들려서,
# "팔을 잘못 벌린 오류"를 잡으려고 어깨가 움직인 바로 그 프레임에서 어깨너비가
# 같이 줄어들어 정규화 기준 자체가 흔들리는 자기모순이 생겼다(실측: arm_offset
# 시나리오 395프레임 중 302프레임에서 어깨너비가 바닥값 아래로 떨어져 거리
# 감점이 통째로 꺼짐 → 잡아야 할 오류를 오히려 못 잡음). 골반너비(23/24, 이미
# _center_on_hip의 원점으로도 씀)는 팔 동작에 거의 영향을 안 받으면서도 카메라
# 거리·몸통 회전(옆모습)에는 어깨너비와 비슷하게 반응해서 훨씬 안정적이다.
_MIN_HIP_WIDTH = 0.03


def _hip_width(coords: np.ndarray) -> float:
    return float(np.linalg.norm(coords[_L_HIP, :2] - coords[_R_HIP, :2]))


def _center_on_hip(coords: np.ndarray) -> np.ndarray:
    """골반 중점을 원점으로 좌표를 이동시킨다.

    코사인 유사도를 화면 절대좌표에 그대로 쓰면, 모든 관절 좌표가 이미
    0~1 사이 양수라 실제 자세가 크게 달라도 벡터 방향이 거의 안 바뀐다
    (실제 영상으로 확인: 팔을 계속 옆으로 벌리고 따라하거나 동작을 반대로
    해도 실시간 score가 0.90~0.996으로 거의 그대로 "Good!"이 나옴, pentagon
    쪽 채점은 같은 데이터로 0~4점까지 정확히 떨어짐). 골반 중점 기준 상대
    좌표로 바꾸면 화면상 위치(translation)는 상쇄되고 실제 자세(shape)
    차이만 코사인 유사도에 반영된다.
    """
    hip_mid = (coords[_L_HIP] + coords[_R_HIP]) / 2.0
    return coords - hip_mid


# Notion "추론" 문서 추후 개선 계획: "관절 가중치 부여 (손목·발목 > 몸통)".
# 실제 영상으로 확인해보니 팔 하나만 계속 잘못 벌린 채 따라해도(오른쪽 어깨/팔꿈치/
# 손목만 어긋나고 나머지 30개 관절은 정확) 코사인 유사도가 0.996→0.983으로 거의
# 안 떨어졌다 — 99개 좌표값 전체를 동등하게 평균 내다 보니 관절 3개의 오차가
# 희석된다. 손목·발목(동작에서 가장 변별력 있는 말단 관절)에 가중치를 줘서
# 이 관절들의 오차가 전체 유사도에 더 크게 반영되게 한다.
_JOINT_WEIGHTS = np.ones(NUM_LANDMARKS, dtype=np.float32)
_WRIST_ANKLE = (15, 16, 27, 28)  # 왼/오 손목, 왼/오 발목
_ELBOW_KNEE = (13, 14, 25, 26)   # 왼/오 팔꿈치, 왼/오 무릎
_JOINT_WEIGHTS[list(_WRIST_ANKLE)] = 3.0
_JOINT_WEIGHTS[list(_ELBOW_KNEE)] = 1.5

# 골반너비 정규화된 최악 관절 오차를 코사인 점수에서 감점할 때 쓰는 계수.
# 실측 스윕 결과(scratchpad formula_sim4.py) k=0.10이 arm_offset 오류는
# 잡으면서(good 0%) 카메라 거리/자연 노이즈/1프레임 지연은 오탐 없이 유지하는
# 지점. (정규화 기준을 어깨너비 → 골반너비로 바꾸면서 오차의 절대적 크기가
# 달라져 계수도 0.15 → 0.10으로 재보정함, _MIN_HIP_WIDTH 주석 참고)
_WORST_JOINT_PENALTY_K = 0.10

# 2026-09-23: 실제 영상(scratchpad 댄스 챌린지 클립)으로 재현해보니, 관절
# 가중치를 줘도 여전히 팔 하나만 계속 잘못 벌린 채 따라하면 395프레임 내내
# score 0.93~0.99, feedback "Good!"만 나왔다. 원인은 33관절×3좌표를 전부
# 하나의 99차원 벡터로 펼쳐서 그 벡터 전체의 방향(코사인 유사도)을 재기
# 때문 — 벡터의 노름(norm) 자체가 32개의 정확한 관절에 의해 지배되어,
# 손목 하나가 완전히 다른 방향을 가리켜도 전체 벡터 방향은 거의 안 바뀐다.
# 그래서 관절별로 각자의 코사인 유사도(골반 기준 방향)를 따로 구한 뒤
# 가중 평균하는 방식으로 바꿨다 — "코사인 유사도"라는 지표 자체는 그대로
# 유지하되(팀 결정대로 RMSE 전면 교체는 하지 않음), 계산 단위를 "전체 벡터
# 1개"에서 "관절 33개 각각"으로 쪼개서 손목 하나만 틀려도 그 관절의 낮은
# 유사도가 평균에 온전히 반영되게 한다.


# boto3 기본값(connect 60s, read 60s, 재시도까지 포함하면 실질적으로 훨씬 김)을 쓰면
# S3가 응답 없을 때 세션 종료 처리(_finalize_session)가 수십 초씩 멈춘다. 실제 서버를
# 띄워 자격증명이 안 맞는 상황을 재현해보니 summary 저장까지 6~7초가 걸렸다.
# 짧게 잡고 재시도 1회로 제한해 실시간 세션 종료 경로가 오래 막히지 않게 한다.
_S3_CLIENT_CONFIG = BotoConfig(
    connect_timeout=2,
    read_timeout=3,
    retries={"max_attempts": 1},
)


async def _load_from_s3(keypoint_path: str) -> np.ndarray:
    """S3에서 .npy keypoint 파일 로드"""
    session = aioboto3.Session(
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )
    async with session.client("s3", config=_S3_CLIENT_CONFIG) as s3:
        resp = await s3.get_object(Bucket=settings.s3_bucket_name, Key=keypoint_path)
        body = await resp["Body"].read()

    arr = np.load(io.BytesIO(body))  # shape: (num_frames, num_joints, 3)
    return arr


async def upload_session_detail(session_id: str, frame_details: list[dict]) -> str:
    """세션 종료 시 프레임별 상세 점수(frame_idx/timestamp_ms/score/worst_joints)를
    S3에 JSON으로 업로드하고 키를 반환한다.

    presigned URL로 클라이언트에 서빙하는 쪽(Spring Boot의 S3ObjectStorage.
    createDownloadPresignedUrl)은 이미 있으므로, 여기서는 파일을 만들어
    올리기만 한다. 업로드 실패 시 세션 종료 처리 자체가 막히면 안 되므로
    예외 처리는 호출부(_finalize_session)에서 감싼다.
    """
    key = f"reports/{session_id}.json"
    session = aioboto3.Session(
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )
    body = json.dumps(frame_details).encode("utf-8")
    async with session.client("s3", config=_S3_CLIENT_CONFIG) as s3:
        await s3.put_object(
            Bucket=settings.s3_bucket_name,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
    return key


async def load_reference_keypoints(
    redis: aioredis.Redis,
    video_id: int,
    keypoint_path: str,
) -> np.ndarray:
    """Redis 캐시 우선, 없으면 S3에서 로드 후 캐시 저장"""
    cache_key = f"ref_kp:{video_id}"
    cached = await redis.get(cache_key)

    if cached:
        arr = np.frombuffer(cached, dtype=np.float32)
        # 원래 shape 복원을 위해 메타 키도 저장
        shape_raw = await redis.get(f"{cache_key}:shape")
        if shape_raw:
            shape = tuple(json.loads(shape_raw))
            arr = arr.reshape(shape)
            _validate_reference_shape(arr)
            return arr

    logger.info("S3에서 reference keypoint 로드: %s", keypoint_path)
    arr = await _load_from_s3(keypoint_path)
    arr = arr.astype(np.float32)
    _validate_reference_shape(arr)

    await redis.set(cache_key, arr.tobytes(), ex=settings.redis_session_ttl)
    await redis.set(
        f"{cache_key}:shape", json.dumps(list(arr.shape)), ex=settings.redis_session_ttl
    )
    return arr


def _validate_reference_shape(arr: np.ndarray) -> None:
    """frame_idx % num_frames에서 ZeroDivisionError가 나지 않도록 프레임 0개인
    reference를 여기서 걸러 로드 단계(4003 close 경로)에서 실패시킨다."""
    if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[1:] != (NUM_LANDMARKS, KEYPOINT_DIM):
        raise ValueError(f"reference keypoints shape이 올바르지 않습니다: {arr.shape}")


def compute_feedback(
    reference: np.ndarray,
    incoming: np.ndarray,
    frame_idx: int,
    timestamp_ms: int | None = None,
    fps: float = 30.0,
) -> dict[str, Any]:
    """
    실시간 keypoint 비교 → 피드백 반환

    reference:    (num_frames, num_joints, 3)  — 강사 영상 기준 [x, y, z]
    incoming:     (num_joints, 3)              — 사용자의 현재 프레임 [x, y, z]
    timestamp_ms: 재생 시작 기준 경과 시간(ms). 제공 시 프레임 드랍 보정에 사용.
    fps:          기준 영상 FPS (세션 메타에서 주입)
    """
    num_frames = reference.shape[0]

    # timestamp_ms가 있으면 경과 시간으로 프레임을 재계산해 드랍 보정.
    # - 클램프(min)가 아니라 순환(%)이어야 재생 시간이 기준 영상 길이를 넘어도
    #   마지막 프레임에 고정되지 않고 처음부터 다시 순환한다.
    # - int() 절삭 대신 round()를 써야 한다: 클라이언트가 보내는 timestamp_ms 자체가
    #   정수 반올림된 값이라, int()로 다시 자르면 오차가 누적돼 30fps 기준
    #   frame_idx가 3프레임마다 한 번꼴로 중복되거나 하나씩 건너뛴다(왕복 시뮬레이션
    #   300프레임 중 반복 100회, 스킵 99회). round()는 이 누적 오차를 0으로 없앤다.
    if timestamp_ms is not None:
        frame_idx = round(timestamp_ms * fps / 1000) % num_frames

    ref_frame = reference[frame_idx % num_frames]

    # 2026-09-23: MediaPipe가 특정 관절을 못 잡으면(화면 밖으로 나감 등) 그
    # 관절만 (0,0,0)으로 오는 경우가 있다 — 실제 몸 좌표는 MediaPipe 정규화
    # 좌표계에서 (0,0,0)(이미지 좌상단 모서리)에 올 일이 거의 없으므로, 이걸
    # "그 관절이 화면 좌상단으로 순간이동했다"는 큰 오차로 취급하면 안 되고
    # "추적 안 됨"으로 보고 그 관절만 채점에서 빼야 한다. 기존에도 pentagon_
    # scoring 쪽엔 "전부 [0,0,0]=사람 전체 미검출" 가드가 있었는데(untracked
    # person), 관절 하나만 빠지는 흔한 경우는 실시간 채점 쪽에 가드가 없었다.
    untracked_mask = np.all(np.abs(incoming) < 1e-6, axis=1)  # (33,) True=추적 안 됨
    joint_weights = _JOINT_WEIGHTS.copy()
    joint_weights[untracked_mask] = 0.0
    if not joint_weights.any():  # 33관절 전부 미검출 — 기존 로직대로 그대로 진행
        joint_weights = _JOINT_WEIGHTS

    # 골반 중점 기준 상대좌표로 변환 후 비교 — 절대좌표 그대로 쓰면 자세가
    # 달라도 코사인 유사도가 거의 안 떨어진다 (자세한 이유는 _center_on_hip 참고).
    ref_coords = _center_on_hip(ref_frame)   # (33, 3) [x, y, z]
    inc_coords = _center_on_hip(incoming)    # (33, 3) [x, y, z]

    # 관절별 코사인 유사도(골반 기준 방향) 후 가중 평균 — 33관절을 하나의
    # 99차원 벡터로 합쳐서 비교하면 안 되는 이유는 _JOINT_WEIGHTS 위 주석 참고.
    # 손목/발목처럼 변별력 있는 말단 관절에는 가중치를 더 준다.
    joint_dots = np.sum(ref_coords * inc_coords, axis=1)
    joint_norms = np.linalg.norm(ref_coords, axis=1) * np.linalg.norm(inc_coords, axis=1)
    joint_cos_sim = joint_dots / (joint_norms + 1e-8)  # (33,) 관절별 -1.0~1.0
    cos_sim = float(np.sum(joint_cos_sim * joint_weights) / np.sum(joint_weights))

    # 2026-09-23: 관절별 코사인 평균만으로는 여전히 부족했다 — 실제 영상으로 재검증
    # 해보니(사용자가 올린 댄스 챌린지 클립) 팔 하나를 계속 잘못 벌린 채 따라해도
    # score가 0.93~0.99, 395프레임 내내 "Good!"이 나왔다. 이유는 손목이 골반에서
    # 멀리 떨어진 관절이라, 위치가 크게 어긋나도 "골반 기준 방향(각도)" 자체는
    # 별로 안 바뀌기 때문 — 코사인 유사도는 각도만 재기 때문에 이런 크기(거리)
    # 오차에는 원래 둔감하다. 그래서 "가장 많이 틀린 관절 하나의 거리 오차"를
    # 코사인 점수에서 그대로 감점하는 항을 추가했다 — 코사인 유사도는 그대로
    # 기본 지표로 두되(팀 결정대로 유지), 방향만으로 못 잡는 오차를 별도 항으로
    # 보정하는 방식.
    #
    # 이 거리 오차는 몸 크기로 정규화해야 한다: 정규화 없이 절대좌표 그대로
    # 쓰면, 사용자가 기준 영상 속 사람보다 카메라에 더 가까이/멀리 서기만 해도
    # (동작 자체는 완벽해도) 관절까지의 거리가 전체적으로 커져서/작아져서
    # 오탐(false negative)이 난다 — 실제 영상으로 확인함(1.3배 확대 시 동작이
    # 똑같은데도 score 평균 0.83, good 비율 72%로 억울하게 깎임). 코사인
    # 유사도는 방향만 보므로 원래 스케일 불변인데, 거리 기반 감점 항만
    # 스케일에 영향을 받고 있었다. 골반너비(23/24)로 나눠서 카메라 거리를
    # 상쇄시킨다(어깨너비 대신 골반너비를 쓰는 이유는 _MIN_HIP_WIDTH 주석 참고).
    raw_ref_scale = _hip_width(ref_coords)
    raw_inc_scale = _hip_width(inc_coords)
    ref_scale = max(raw_ref_scale, _MIN_HIP_WIDTH)
    inc_scale = max(raw_inc_scale, _MIN_HIP_WIDTH)
    joint_errors = np.linalg.norm(
        ref_coords / ref_scale - inc_coords / inc_scale, axis=1
    )  # (33,) 골반너비 기준 정규화된 거리 오차 — 카메라 거리에 무관
    # 추적 안 된 관절은 오차도 "가짜로 크게" 나오므로 worst_joints 후보에서 제외.
    ranked = np.argsort(joint_errors)[::-1]
    worst_joints = [int(j) for j in ranked if not untracked_mask[j]][:3]

    # 정규화하면서 오차 값의 스케일 자체가 바뀌어서 감점 계수를 1.0 그대로
    # 쓰면 모든 시나리오가 -1.0(클리핑 하한)으로 몰리는 과보정이 발생했다.
    # 실제 영상으로 k값을 스윕(scratchpad formula_sim4.py)해서 재보정함 — 값은
    # 아래 _WORST_JOINT_PENALTY_K 주석 참고.
    #
    # 몸을 완전히 돌려 골반까지 거의 겹쳐 보이는 극단적인 경우(카메라를 거의
    # 등지는 수준)에는 골반너비도 바닥값까지 깎일 수 있다. 이때는 스케일
    # 추정 자체를 못 믿는 것이므로 거리 기반 감점을 걸지 않고 방향(코사인)만
    # 으로 채점한다 — pentagon_scoring이 퇴화 케이스를 계산 스킵하는 것과
    # 같은 맥락.
    scale_reliable = (
        raw_ref_scale >= _MIN_HIP_WIDTH and raw_inc_scale >= _MIN_HIP_WIDTH
    )
    has_tracked_joint = bool((~untracked_mask).any())
    if scale_reliable and has_tracked_joint:
        worst_joint_penalty = _WORST_JOINT_PENALTY_K * float(
            np.max((joint_errors * (joint_weights / joint_weights.max()))[~untracked_mask])
        )
    else:
        worst_joint_penalty = 0.0
    score = max(-1.0, cos_sim - worst_joint_penalty)

    feedback_text = _score_to_message(score)

    return {
        "type": "feedback",  # FE(use-ai-feedback-socket.ts)가 msg.type으로 메시지 종류를 구분함
        "score": round(score, 4),
        "message": feedback_text,  # FE의 FeedbackMessage.message 필드명에 맞춤 (구 필드명: feedback)
        "frame_idx": frame_idx,
        "worst_joints": worst_joints,  # 가장 틀린 관절 인덱스
    }


def _score_to_message(score: float) -> str:
    if score >= settings.feedback_threshold_good:
        return "Good!"
    if score >= settings.feedback_threshold_bad:
        return "조금 더 정확하게 따라해 보세요."
    return "동작이 많이 다릅니다. 다시 시도해 보세요."
