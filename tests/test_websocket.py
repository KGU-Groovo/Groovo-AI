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
        {"sub": "no-such-session", "userId": 1},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        msg = ws.receive_json()
        assert msg == {"error": "session not found: no-such-session"}

        with pytest.raises(Exception) as exc_info:
            ws.receive_json()
        assert getattr(exc_info.value, "code", None) == 4002

    # 애초에 존재하지 않던 세션이므로 finalize가 좀비 키를 만들면 안 된다.
    exists = asyncio.run(fake_redis.exists("session:no-such-session"))
    assert exists == 0


def test_malformed_session_data_closes_with_4002(client, fake_redis):
    # 해시는 존재하지만 필수 필드(video_id, keypoint_path 등)가 빠진 손상된 세션 데이터
    asyncio.run(fake_redis.hset("session:test-session-1", mapping={"status": "active"}))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        msg = ws.receive_json()
        assert "error" in msg

        with pytest.raises(Exception) as exc_info:
            ws.receive_json()
        assert getattr(exc_info.value, "code", None) == 4002

    # 세션 키 자체는 존재했으므로(데이터만 깨짐), active로 방치되지 않고
    # finished로 갱신되어야 한다.
    session = asyncio.run(fake_redis.hgetall("session:test-session-1"))
    assert session[b"status"] == b"finished"


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

    # 세션은 이미 Redis에 존재했으므로(2단계 통과), active로 방치되지 않고
    # finished로 갱신되어야 한다.
    session = asyncio.run(fake_redis.hgetall("session:test-session-1"))
    assert session[b"status"] == b"finished"


async def _seed_session_only(redis):
    session_fields = {
        "user_id": "1",
        "video_id": "999",
        "keypoint_path": "keypoints/video_999.npy",
        "fps": "30.0",
        "status": "active",
        "started_at": "0",
    }
    await redis.hset("session:test-session-1", mapping=session_fields)


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


def test_session_finalized_with_summary_on_normal_disconnect(client, fake_redis, fake_s3_upload):
    """정상 종료 시 세션 상태가 finished로 바뀌고, summary가 저장돼야 한다."""
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready
        ws.send_json({"frame_idx": 0, "timestamp_ms": 0, "keypoints": reference_frame(0)})
        first = ws.receive_json()
        ws.send_json({"frame_idx": 1, "timestamp_ms": 33, "keypoints": reference_frame(1)})
        second = ws.receive_json()
    # `with` 블록을 벗어나면 클라이언트가 연결을 닫는다 (WebSocketDisconnect 트리거).

    session = asyncio.run(fake_redis.hgetall("session:test-session-1"))
    assert session[b"status"] == b"finished"
    assert b"finished_at" in session

    raw_summary = asyncio.run(fake_redis.get("session:test-session-1:summary"))
    assert raw_summary is not None
    summary = json.loads(raw_summary)
    assert summary["frame_count"] == 2
    expected_avg = round((first["score"] + second["score"]) / 2, 4)
    assert summary["average_score"] == expected_avg
    assert summary["detail_path"] == "reports/test-session-1.json"

    # 업로드된 프레임별 상세 데이터도 검증
    assert len(fake_s3_upload) == 1
    uploaded_session_id, uploaded_frames = fake_s3_upload[0]
    assert uploaded_session_id == "test-session-1"
    assert [f["score"] for f in uploaded_frames] == [first["score"], second["score"]]
    assert uploaded_frames[0]["frame_idx"] == 0
    assert uploaded_frames[0]["worst_joints"] == first["worst_joints"]


def test_detail_upload_failure_still_saves_average_score(client, fake_redis, monkeypatch):
    """S3 업로드가 실패해도 평균 점수는 저장되고, detail_path만 None이어야 한다."""
    async def _raise(*args, **kwargs):
        raise RuntimeError("S3 unreachable in test")

    monkeypatch.setattr("app.routers.websocket.upload_session_detail", _raise)

    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready
        ws.send_json({"frame_idx": 0, "timestamp_ms": 0, "keypoints": reference_frame(0)})
        ws.receive_json()

    raw_summary = asyncio.run(fake_redis.get("session:test-session-1:summary"))
    summary = json.loads(raw_summary)
    assert summary["frame_count"] == 1
    assert summary["detail_path"] is None


def test_finalize_does_not_resurrect_already_expired_session(client, fake_redis):
    """세션 TTL이 이미 만료된 뒤 종료되면, HSET으로 TTL 없는 좀비 키를
    새로 만들지 않아야 한다."""
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready
        ws.send_json({"frame_idx": 0, "timestamp_ms": 0, "keypoints": reference_frame(0)})
        ws.receive_json()
        # 세션 종료 처리 직전에 세션 키가 만료된 상황을 흉내낸다.
        asyncio.run(fake_redis.delete("session:test-session-1"))

    exists = asyncio.run(fake_redis.exists("session:test-session-1"))
    assert exists == 0  # 좀비 키로 되살아나지 않아야 함


def test_no_summary_when_no_frames_processed(client, fake_redis):
    """프레임을 하나도 못 받고 끝나면 summary는 안 남아야 한다."""
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready

    session = asyncio.run(fake_redis.hgetall("session:test-session-1"))
    assert session[b"status"] == b"finished"

    raw_summary = asyncio.run(fake_redis.get("session:test-session-1:summary"))
    assert raw_summary is None


def test_summary_ttl_matches_session_ttl(client, fake_redis):
    asyncio.run(seed_session_and_reference(fake_redis))
    asyncio.run(fake_redis.expire("session:test-session-1", 1800))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready
        ws.send_json({"frame_idx": 0, "timestamp_ms": 0, "keypoints": reference_frame(0)})
        ws.receive_json()

    summary_ttl = asyncio.run(fake_redis.ttl("session:test-session-1:summary"))
    assert 0 < summary_ttl <= 1800


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


def test_no_pentagon_scores_under_30_frames(client, fake_redis):
    """pentagon_scoring은 30프레임 미만이면 채점 불가하므로 None이어야 한다."""
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready
        for i in range(5):
            ws.send_json(
                {"frame_idx": i, "timestamp_ms": i * 33, "keypoints": reference_frame(i)}
            )
            ws.receive_json()

    raw_summary = asyncio.run(fake_redis.get("session:test-session-1:summary"))
    summary = json.loads(raw_summary)
    assert summary["pentagon_scores"] is None


def test_pentagon_scores_included_after_30_frames(client, fake_redis):
    """30프레임 이상 모이면 pentagon_scoring 오각형 점수가 summary에 담겨야 한다.

    31프레임을 보내는 이유: 정확히 30번째 프레임에서 트리거된 to_thread 채점이
    끝나기 전에 테스트 클라이언트가 바로 연결을 닫아버리면(트리거 프레임 == 마지막
    프레임) asyncio.to_thread await가 연결 종료와 경합할 수 있어, 트리거 이후
    프레임을 하나 더 보내 채점이 끝날 시간을 준다.
    """
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready
        for i in range(31):
            ws.send_json(
                {"frame_idx": i, "timestamp_ms": i * 33, "keypoints": reference_frame(i)}
            )
            ws.receive_json()

    raw_summary = asyncio.run(fake_redis.get("session:test-session-1:summary"))
    summary = json.loads(raw_summary)
    pentagon = summary["pentagon_scores"]
    assert pentagon is not None
    assert pentagon["window_count"] == 1
    assert set(pentagon["scores"]) == {"accuracy", "detail", "balance", "timing", "rhythm"}
    assert 0 <= pentagon["final_score"] <= 100


def test_pentagon_scoring_failure_keeps_session_and_realtime_feedback_alive(
    client, fake_redis, monkeypatch
):
    """pentagon 채점이 예외를 던져도 실시간 피드백/세션 종료 처리는 영향받지 않아야 한다."""

    def _raise(*args, **kwargs):
        raise RuntimeError("pentagon scoring exploded")

    monkeypatch.setattr("app.routers.websocket.score_window", _raise)

    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready
        for i in range(31):
            ws.send_json(
                {"frame_idx": i, "timestamp_ms": i * 33, "keypoints": reference_frame(i)}
            )
            resp = ws.receive_json()
            assert "score" in resp  # 실시간 피드백은 정상 유지

    session = asyncio.run(fake_redis.hgetall("session:test-session-1"))
    assert session[b"status"] == b"finished"

    raw_summary = asyncio.run(fake_redis.get("session:test-session-1:summary"))
    summary = json.loads(raw_summary)
    assert summary["frame_count"] == 31
    assert summary["pentagon_scores"] is None  # 계산 실패한 윈도우는 결과에 안 들어감


def test_pentagon_scoring_survives_reference_shorter_than_session(client, fake_redis):
    """기준 영상이 세션 길이보다 짧아도(실사용에선 드물지만) 크래시 없이 넘어가야 한다.

    keypoint_service.compute_feedback의 frame_idx 보정은 `min(...)`으로 클램프만
    하고 순환(wrap)하지 않는다 (app/services/keypoint_service.py:117, 기존 코드,
    이번 pentagon 연동 대상 아님). 즉 재생 시간이 기준 영상 길이를 넘어가면
    frame_idx가 마지막 프레임에 고정된다. 실사용에선 기준 영상이 세션보다 훨씬
    기므로(3분/2700프레임) 거의 발생하지 않지만, 발생해도 pentagon 채점이 예외 없이
    (점수 품질은 별개로) 끝까지 도는지는 확인해둔다.
    """
    asyncio.run(
        seed_session_and_reference(fake_redis, num_frames=10)
    )
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready
        for i in range(31):
            ws.send_json(
                {
                    "frame_idx": i,
                    "timestamp_ms": i * 33,
                    "keypoints": reference_frame(i, num_frames=10),
                }
            )
            resp = ws.receive_json()
            assert "score" in resp  # 실시간 경로는 크래시 없이 계속 응답

    raw_summary = asyncio.run(fake_redis.get("session:test-session-1:summary"))
    summary = json.loads(raw_summary)
    assert summary["frame_count"] == 31
    # 크래시 없이 결과가 나오기만 하면 됨 (frame_idx가 고정돼 점수 품질 자체는 보장 안 함)
    assert summary["pentagon_scores"] is not None


def test_pentagon_scores_run_every_30_frames_sliding(client, fake_redis):
    """30프레임마다 슬라이딩 윈도우로 보조 채점이 반복 실행돼야 한다 (60번째에서 2번째 실행)."""
    asyncio.run(seed_session_and_reference(fake_redis))
    token = make_token()

    with client.websocket_connect(f"/ws/analyze?token={token}") as ws:
        ws.receive_json()  # ready
        for i in range(65):
            ws.send_json(
                {"frame_idx": i, "timestamp_ms": i * 33, "keypoints": reference_frame(i)}
            )
            ws.receive_json()

    raw_summary = asyncio.run(fake_redis.get("session:test-session-1:summary"))
    summary = json.loads(raw_summary)
    pentagon = summary["pentagon_scores"]
    assert pentagon is not None
    assert pentagon["window_count"] == 2
