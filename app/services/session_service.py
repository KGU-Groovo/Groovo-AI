import json
from dataclasses import dataclass

import redis.asyncio as aioredis


@dataclass
class SessionInfo:
    user_id: int
    video_id: int
    keypoint_path: str  # S3 key (e.g. "keypoints/video_123.npy")
    fps: float = 30.0   # 기준 영상 FPS — timestamp_ms 기반 프레임 보정에 사용


async def get_session(redis: aioredis.Redis, session_id: str) -> SessionInfo:
    """Redis에서 Spring Boot가 저장한 세션 메타데이터 조회"""
    raw = await redis.get(f"session:{session_id}")
    if raw is None:
        raise LookupError(f"session not found: {session_id}")

    try:
        data = json.loads(raw)
        return SessionInfo(
            user_id=data["user_id"],
            video_id=data["video_id"],
            keypoint_path=data["keypoint_path"],
            fps=float(data.get("fps", 30.0)),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise LookupError(f"invalid session data: {session_id}") from e
