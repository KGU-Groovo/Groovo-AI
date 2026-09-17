"""Spring Boot(Groovo-BE)가 실제로 쓰는 데이터 형식과의 연동 회귀 테스트.

- 세션 메타데이터: SessionRedisStore가 Redis Hash(HSET)로 저장함 (JSON 문자열 아님).
- ws_token: JwtProvider.create(subject=session_id, claims={"userId": ..., "videoId": ...})로
  발급되므로 session_id는 `sub`, user_id는 `userId` claim에 있음.
"""
import asyncio

import fakeredis.aioredis
import pytest
from jose import jwt

from app.config import settings
from app.services.session_service import get_session
from app.services.token_service import verify_ws_token


def test_verify_ws_token_reads_sub_and_camelcase_user_id():
    token = jwt.encode(
        {"sub": "sid-123", "userId": 42, "videoId": 7},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    payload = verify_ws_token(token)

    assert payload.session_id == "sid-123"
    assert payload.user_id == 42


def test_verify_ws_token_rejects_old_snake_case_claims():
    """구 포맷(session_id/user_id claim)은 실제 BE가 발급하지 않으므로 거부되어야 한다."""
    token = jwt.encode(
        {"session_id": "sid-123", "user_id": 42},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(ValueError):
        verify_ws_token(token)


def test_get_session_reads_redis_hash_written_by_spring_boot():
    async def _run():
        redis = fakeredis.aioredis.FakeRedis()
        await redis.hset(
            "session:sid-123",
            mapping={
                "user_id": "42",
                "video_id": "7",
                "keypoint_path": "keypoints/video_7.npy",
                "fps": "30.0",
                "status": "active",
                "started_at": "1234567890",
            },
        )

        session = await get_session(redis, "sid-123")

        assert session.user_id == 42
        assert session.video_id == 7
        assert session.keypoint_path == "keypoints/video_7.npy"
        assert session.fps == 30.0

    asyncio.run(_run())


def test_get_session_rejects_non_numeric_fields():
    """user_id/video_id가 숫자로 변환 안 되면(KeyError가 아니라 ValueError)도
    깔끔하게 LookupError(→ 4002)로 처리되어야 한다."""
    async def _run():
        redis = fakeredis.aioredis.FakeRedis()
        await redis.hset(
            "session:sid-123",
            mapping={
                "user_id": "not-a-number",
                "video_id": "7",
                "keypoint_path": "keypoints/video_7.npy",
                "fps": "30.0",
            },
        )

        with pytest.raises(LookupError):
            await get_session(redis, "sid-123")

    asyncio.run(_run())


def test_get_session_no_longer_crashes_on_wrongtype_get():
    """예전엔 redis.get()으로 Hash 키를 읽어 WRONGTYPE 에러로 죽었다. HGETALL로 읽어야 한다."""
    async def _run():
        redis = fakeredis.aioredis.FakeRedis()
        await redis.hset("session:sid-123", mapping={"user_id": "1"})

        # plain GET on a hash key must fail — confirms the bug this test guards against
        with pytest.raises(Exception):
            await redis.get("session:sid-123")

        # but get_session (HGETALL-based) should raise a clean LookupError instead,
        # not the raw redis WRONGTYPE error, since required fields are still missing
        with pytest.raises(LookupError):
            await get_session(redis, "sid-123")

    asyncio.run(_run())
