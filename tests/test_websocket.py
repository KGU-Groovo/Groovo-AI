import asyncio
import json

import numpy as np
import pytest
from jose import jwt

import app.routers.websocket as ws_module
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


def test_binary_frame_keeps_session_alive(client, fake_redis):
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready

        ws.send_bytes(b"\x00\x01\x02")
        resp = ws.receive_json()
        assert "error" in resp

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


def test_malformed_session_data_closes_with_4002(client, fake_redis):
    # 비-JSON 문자열 등, 파싱 자체가 깨진 손상된 세션 데이터
    asyncio.run(fake_redis.set("session:test-session-1", "not even json"))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        msg = ws.receive_json()
        assert "error" in msg

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
def test_warn_after_silence(client, fake_redis, monkeypatch):
    """수신 없이 WARN_SEC 이상 지나면 경고만 오고 recommend_pause는 아직 없어야 한다."""
    monkeypatch.setattr(ws_module, "_WARN_SEC", 0.1)
    monkeypatch.setattr(ws_module, "_PAUSE_SEC", 5.0)
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready

        ws.send_json({"frame_idx": 0, "timestamp_ms": 0, "keypoints": reference_frame(0)})
        ws.receive_json()  # 첫 프레임 정상 응답

        resp = ws.receive_json()  # 이후 아무것도 안 보내면 WARN_SEC 뒤 경고가 와야 함
        assert resp.get("warning")
        assert "recommend_pause" not in resp


@pytest.mark.timeout(10)
def test_recommend_pause_after_extended_silence(client, fake_redis, monkeypatch):
    """수신 없이 PAUSE_SEC 이상 지나면 recommend_pause=True와 마지막 점수가 와야 한다."""
    monkeypatch.setattr(ws_module, "_WARN_SEC", 0.1)
    monkeypatch.setattr(ws_module, "_PAUSE_SEC", 0.3)
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready

        ws.send_json({"frame_idx": 0, "timestamp_ms": 0, "keypoints": reference_frame(0)})
        first = ws.receive_json()  # 첫 프레임 정상 응답

        resp = ws.receive_json()
        while not resp.get("recommend_pause"):
            resp = ws.receive_json()

        assert resp["recommend_pause"] is True
        assert resp["score"] == first["score"]
        assert resp["frame_idx"] == first["frame_idx"]


def test_empty_reference_closes_with_4003(client, fake_redis):
    """캐시된 reference keypoint가 0프레임이면 ZeroDivisionError로 죽지 않고 4003으로 닫혀야 한다."""
    asyncio.run(_seed_session_only(fake_redis))
    video_id = 999
    cache_key = f"ref_kp:{video_id}"
    empty_ref = np.zeros((0, 33, 3), dtype=np.float32)
    asyncio.run(fake_redis.set(cache_key, empty_ref.tobytes()))
    asyncio.run(fake_redis.set(f"{cache_key}:shape", json.dumps(list(empty_ref.shape))))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        msg = ws.receive_json()
        assert msg == {"error": "keypoint 로드 실패"}

        with pytest.raises(Exception) as exc_info:
            ws.receive_json()
        assert getattr(exc_info.value, "code", None) == 4003


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
