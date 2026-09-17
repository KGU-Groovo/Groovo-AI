import json

import fakeredis.aioredis
import numpy as np
import pytest
from jose import jwt
from starlette.testclient import TestClient

import app.redis_client as redis_client_module
from app.config import settings
from app.main import app

NUM_FRAMES = 60


@pytest.fixture
def fake_redis():
    """실제 Redis 대신 in-process fake로 교체. Docker/네트워크 불필요."""
    fake = fakeredis.aioredis.FakeRedis()
    redis_client_module._redis = fake
    yield fake
    redis_client_module._redis = None


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def make_reference_sequence(num_frames: int = NUM_FRAMES) -> np.ndarray:
    """33관절 sine-wave 동작 시퀀스. seed_redis용 기준 데이터."""
    t = np.linspace(0, 4 * np.pi, num_frames)
    ref = np.zeros((num_frames, 33, 3), dtype=np.float32)
    for j in range(33):
        phase = j * 0.1
        ref[:, j, 0] = 0.5 + 0.05 * np.sin(t + phase)
        ref[:, j, 1] = 0.5 + 0.05 * np.cos(t + phase)
        ref[:, j, 2] = 0.1 * np.sin(t * 0.5 + phase)
    return ref


def reference_frame(frame_idx: int, num_frames: int = NUM_FRAMES) -> list:
    """make_reference_sequence와 동일한 공식으로 frame_idx 시점 좌표 재현."""
    t = frame_idx / num_frames * 4 * np.pi
    frame = []
    for j in range(33):
        phase = j * 0.1
        x = 0.5 + 0.05 * np.sin(t + phase)
        y = 0.5 + 0.05 * np.cos(t + phase)
        z = 0.1 * np.sin(t * 0.5 + phase)
        frame.append([float(x), float(y), float(z)])
    return frame


async def seed_session_and_reference(
    redis,
    session_id: str = "test-session-1",
    video_id: int = 42,
    user_id: int = 1,
    fps: float = 30.0,
    num_frames: int = NUM_FRAMES,
) -> None:
    """S3 호출 없이 ref_kp 캐시 히트를 유도 + session 메타데이터 저장."""
    ref = make_reference_sequence(num_frames)
    cache_key = f"ref_kp:{video_id}"
    await redis.set(cache_key, ref.tobytes())
    await redis.set(f"{cache_key}:shape", json.dumps(list(ref.shape)))

    # Spring Boot(SessionRedisStore)와 동일하게 Hash(HSET)로 저장한다.
    session_fields = {
        "user_id": str(user_id),
        "video_id": str(video_id),
        "keypoint_path": f"keypoints/video_{video_id}.npy",
        "fps": str(fps),
        "status": "active",
        "started_at": "0",
    }
    await redis.hset(f"session:{session_id}", mapping=session_fields)


def make_token(session_id: str = "test-session-1", user_id: int = 1) -> str:
    """Spring Boot(JwtProvider.create)와 동일하게 subject=session_id, userId claim으로 발급."""
    return jwt.encode(
        {"sub": session_id, "userId": user_id},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
