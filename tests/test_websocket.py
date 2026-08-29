import asyncio
import json

import numpy as np
import pytest
from jose import jwt

from app.config import settings
from tests.conftest import make_token, reference_frame, seed_session_and_reference


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_good_frame_gives_high_score(client, fake_redis):
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ready = ws.receive_json()
        assert ready == {"status": "ready", "video_id": 42}

        ws.send_json({"frame_idx": 0, "timestamp_ms": 0, "keypoints": reference_frame(0)})
        resp = ws.receive_json()
        assert resp["score"] > 0.99
        assert resp["feedback"] == "Good!"


def test_bad_frame_gives_low_score(client, fake_redis):
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready

        rng = np.random.default_rng(0)
        noisy = (rng.random((33, 3)) * 2 - 1).tolist()
        ws.send_json({"frame_idx": 0, "timestamp_ms": 0, "keypoints": noisy})
        resp = ws.receive_json()
        assert resp["score"] < 0.5


def test_missing_keypoints_field_keeps_session_alive(client, fake_redis):
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready

        ws.send_json({"frame_idx": 0, "timestamp_ms": 0})
        resp = ws.receive_json()
        assert resp == {"error": "keypoints 필드 누락"}

        # 세션이 살아있는지 정상 프레임으로 재확인
        ws.send_json({"frame_idx": 1, "timestamp_ms": 33, "keypoints": reference_frame(1)})
        resp2 = ws.receive_json()
        assert "score" in resp2


def test_malformed_json_keeps_session_alive(client, fake_redis):
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready

        ws.send_text("this is not json")
        resp = ws.receive_json()
        assert resp == {"error": "잘못된 JSON 형식입니다"}

        ws.send_json({"frame_idx": 1, "timestamp_ms": 33, "keypoints": reference_frame(1)})
        resp2 = ws.receive_json()
        assert "score" in resp2


def test_invalid_token_closes_with_4001(client, fake_redis):
    bad_token = "this-is-not-a-valid-jwt"

    with client.websocket_connect(f"/ws/analyze?token={bad_token}") as ws:
        msg = ws.receive_json()
        assert "error" in msg

        with pytest.raises(Exception) as exc_info:
            ws.receive_json()
        # starlette TestClient raises WebSocketDisconnect with .code
        assert getattr(exc_info.value, "code", None) == 4001


def test_unknown_session_closes_with_4002(client, fake_redis):
    token = jwt.encode(
        {"session_id": "no-such-session", "user_id": 1},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        msg = ws.receive_json()
        assert msg == {"error": "session not found: no-such-session"}

        with pytest.raises(Exception) as exc_info:
            ws.receive_json()
        assert getattr(exc_info.value, "code", None) == 4002


def test_reference_load_failure_closes_with_4003(client, fake_redis, monkeypatch):
    # 세션은 있지만 ref_kp 캐시도, S3도 없는 상태 -> keypoint 로드 실패
    asyncio.run(
        _seed_session_only(fake_redis)
    )
    token = make_token()

    async def _raise(*args, **kwargs):
        raise RuntimeError("S3 unreachable in test")

    monkeypatch.setattr("app.services.keypoint_service._load_from_s3", _raise)

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        msg = ws.receive_json()
        assert msg == {"error": "keypoint 로드 실패"}

        with pytest.raises(Exception) as exc_info:
            ws.receive_json()
        assert getattr(exc_info.value, "code", None) == 4003


async def _seed_session_only(redis):
    session_value = {
        "user_id": 1,
        "video_id": 999,
        "keypoint_path": "keypoints/video_999.npy",
        "fps": 30.0,
        "status": "active",
        "started_at": 0,
    }
    await redis.set("session:test-session-1", json.dumps(session_value))


@pytest.mark.timeout(10)
def test_wrong_joint_count_keeps_session_alive(client, fake_redis):
    """관절 개수가 33개가 아닐 때 세션이 끊기지 않고 에러만 받는지 확인."""
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready

        bad_kp = [[0.1, 0.1, 0.1]] * 5  # 33개가 아니라 5개
        ws.send_json({"frame_idx": 0, "timestamp_ms": 0, "keypoints": bad_kp})
        resp = ws.receive_json()
        assert "error" in resp

        ws.send_json({"frame_idx": 1, "timestamp_ms": 33, "keypoints": reference_frame(1)})
        resp2 = ws.receive_json()
        assert "score" in resp2


@pytest.mark.timeout(10)
def test_ragged_keypoints_keeps_session_alive(client, fake_redis):
    """관절마다 좌표 개수가 다른 ragged 배열이 와도 세션이 끊기지 않는지 확인."""
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready

        ragged_kp = [[0.1, 0.1, 0.1]] * 32 + [[0.1, 0.1]]  # 마지막 관절만 좌표 2개
        ws.send_json({"frame_idx": 0, "timestamp_ms": 0, "keypoints": ragged_kp})
        resp = ws.receive_json()
        assert "error" in resp

        ws.send_json({"frame_idx": 1, "timestamp_ms": 33, "keypoints": reference_frame(1)})
        resp2 = ws.receive_json()
        assert "score" in resp2
