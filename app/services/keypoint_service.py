import io
import json
import logging
from pathlib import Path
from typing import Any

import aioboto3
import numpy as np
import redis.asyncio as aioredis

from app.config import settings
from src.normalization import normalize_pose_frame

logger = logging.getLogger(__name__)

# MediaPipe Pose 기준: 33 landmarks × [x, y, z]
NUM_LANDMARKS = 33
KEYPOINT_DIM = 3
TIME_ALIGNMENT_WINDOW_FRAMES = 6
MIN_POSE_SCALE = 1e-6


async def _load_from_s3(keypoint_path: str) -> np.ndarray:
    """S3에서 .npy keypoint 파일 로드"""
    session = aioboto3.Session(
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )
    async with session.client("s3") as s3:
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
    async with session.client("s3") as s3:
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
    """로컬 개발 파일 우선, 그 외에는 Redis 캐시와 S3를 사용한다."""
    local_path = Path(keypoint_path)
    if local_path.is_file():
        arr = np.load(local_path).astype(np.float32)
        _validate_reference_shape(arr)
        return arr

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


def _cosine_similarity(reference_frame: np.ndarray, incoming: np.ndarray) -> float:
    reference_flat = reference_frame.flatten()
    incoming_flat = incoming.flatten()
    return float(
        np.dot(reference_flat, incoming_flat)
        / (np.linalg.norm(reference_flat) * np.linalg.norm(incoming_flat) + 1e-8)
    )


def _comparison_pose_pair(reference_frame: np.ndarray, incoming: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference_width = np.linalg.norm(reference_frame[11] - reference_frame[12])
    incoming_width = np.linalg.norm(incoming[11] - incoming[12])
    if reference_width < MIN_POSE_SCALE or incoming_width < MIN_POSE_SCALE:
        return reference_frame, incoming
    return normalize_pose_frame(reference_frame), normalize_pose_frame(incoming)


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

    expected_frame_idx = frame_idx % num_frames
    candidate_frame_indices = [expected_frame_idx]
    for offset in range(1, TIME_ALIGNMENT_WINDOW_FRAMES + 1):
        candidate_frame_indices.extend(
            [
                (expected_frame_idx - offset) % num_frames,
                (expected_frame_idx + offset) % num_frames,
            ]
        )

    # 카메라/포즈 추론 지연은 짧은 구간의 동작 시점 차이로 나타난다.
    # 기준 시점을 먼저 넣어 동점인 정지 포즈에서는 원래 재생 위치를 유지한다.
    frame_idx = max(
        candidate_frame_indices,
        key=lambda candidate_idx: _cosine_similarity(
            *_comparison_pose_pair(reference[candidate_idx], incoming)
        ),
    )
    ref_frame, normalized_incoming = _comparison_pose_pair(reference[frame_idx], incoming)

    ref_coords = ref_frame   # (33, 3) [x, y, z]
    inc_coords = normalized_incoming

    # 코사인 유사도
    cos_sim = _cosine_similarity(ref_coords, inc_coords)

    # 관절별 거리 오차 (정규화된 좌표 기준)
    joint_errors = np.linalg.norm(ref_coords - inc_coords, axis=1)
    worst_joints = np.argsort(joint_errors)[::-1][:3].tolist()

    feedback_text = _score_to_message(cos_sim)

    return {
        "score": round(cos_sim, 4),
        "feedback": feedback_text,
        "frame_idx": frame_idx,
        "timestamp_ms": timestamp_ms,
        "worst_joints": worst_joints,  # 가장 틀린 관절 인덱스
    }


def _score_to_message(score: float) -> str:
    if score >= settings.feedback_threshold_good:
        return "Good!"
    if score >= settings.feedback_threshold_bad:
        return "조금 더 정확하게 따라해 보세요."
    return "동작이 많이 다릅니다. 다시 시도해 보세요."
