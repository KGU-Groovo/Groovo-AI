from dataclasses import dataclass

import redis.asyncio as aioredis


@dataclass
class SessionInfo:
    user_id: int
    video_id: int
    keypoint_path: str  # S3 key (e.g. "keypoints/video_123.npy")
    fps: float = 30.0   # 기준 영상 FPS — timestamp_ms 기반 프레임 보정에 사용


def _decode(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


async def get_session(redis: aioredis.Redis, session_id: str) -> SessionInfo:
    """Redis에서 Spring Boot가 저장한 세션 메타데이터 조회.

    Spring Boot(SessionRedisStore)는 session:{id}를 JSON 문자열이 아니라
    Redis Hash(HSET)로 저장한다. 여기서 GET을 쓰면 WRONGTYPE 에러로 죽으므로
    HGETALL로 읽어야 한다.
    """
    raw = await redis.hgetall(f"session:{session_id}")
    if not raw:
        raise LookupError(f"session not found: {session_id}")

    data = {_decode(k): _decode(v) for k, v in raw.items()}

    try:
        return SessionInfo(
            user_id=int(data["user_id"]),
            video_id=int(data["video_id"]),
            keypoint_path=data["keypoint_path"],
            fps=float(data.get("fps", 30.0)),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise LookupError(f"invalid session data: {session_id}") from e
