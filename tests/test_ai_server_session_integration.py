"""integration/ai-server-session 통합 검증.

feature/ai-server(ws_token/Redis 세션/종료 처리/PR #7 채점)와
integration/ai-server-fe(시간 정렬/DCA/pentagon 응답 필드)를 합친 뒤,
두 흐름이 BE 세션 기반 연결 하나에서 함께 동작하는지 확인한다.

외부 파일(.npy reference, DCA checkpoint) 없이 실행된다:
- reference는 conftest의 sine-wave 시퀀스를 fake Redis 캐시에 넣어 쓴다.
- DCA는 가짜 모델(FakeDcaModel) 또는 학습되지 않은 모델 구조(체크포인트 없이
  create_dca_model()로 메모리에서만 생성)로 응답 계약만 검증한다.
"""

import asyncio
import json

import numpy as np
import pytest
import torch

import app.routers.websocket as websocket_router
import app.services.dca_model_service as dca_model_service
from app.config import settings
from app.services.keypoint_service import compute_feedback
from tests.conftest import (
    make_reference_sequence,
    make_token,
    reference_frame,
    seed_session_and_reference,
)


class FakeDcaModel:
    """DcaModelRunner.predict와 같은 반환 형태를 흉내낸다."""

    def __init__(self):
        self.calls = 0

    def predict(self, reference_window, user_window):
        assert reference_window.shape == (30, 33, 3)
        assert user_window.shape == (30, 33, 3)
        self.calls += 1
        return {
            "score_norm": 0.87,
            "frontend_payload": {"score_100": 87.0, "highlight_joints": [11, 13, 15]},
        }


def _send_frames(ws, count: int) -> list[dict]:
    responses = []
    for i in range(count):
        ws.send_json({"frame_idx": i, "timestamp_ms": round(i * 1000 / 30), "keypoints": reference_frame(i)})
        responses.append(ws.receive_json())
    return responses


# ── BE 세션 기반 기본 연결 ────────────────────────────────────────────────


def test_token_session_flow_returns_frame_score_pentagon_and_dca(client, fake_redis, monkeypatch):
    """ws_token → Redis 세션 → reference 로드 → 실시간 채점 + 30프레임 보조 결과."""
    asyncio.run(seed_session_and_reference(fake_redis))
    fake_dca = FakeDcaModel()
    monkeypatch.setattr(websocket_router, "get_dca_model_runner", lambda: fake_dca)

    with client.websocket_connect(f"/ws/analyze?token={make_token()}") as ws:
        assert ws.receive_json() == {"status": "ready", "video_id": 42}
        responses = _send_frames(ws, 31)

    # 매 프레임 공통 필드 (FE PR #5 + feature/ai-server 필드 모두 유지)
    for resp in responses:
        assert resp["type"] == "feedback"
        assert set(resp) >= {"score", "feedback", "message", "frame_idx", "worst_joints"}
        assert resp["feedback"] == resp["message"]
        assert resp["score"] > 0.99  # 기준 동작 그대로 → 프레임 점수는 만점 근처

    window_resp = responses[29]  # 30번째 프레임에서 pentagon/DCA 보조 결과 포함
    assert window_resp["pentagon_scores"]["final_score"] == window_resp["pentagon_scores"]["scores"]["accuracy"]
    assert set(window_resp["pentagon_scores"]["scores"]) == {"accuracy", "detail", "balance", "timing", "rhythm"}
    assert window_resp["dca"] == {"score_100": 87.0, "highlight_joints": [11, 13, 15]}
    # score는 pentagon accuracy로 덮어쓰지 않고 항상 프레임 점수다.
    assert window_resp["rule_score"] == window_resp["score"]
    assert window_resp["score"] > 0.99
    assert fake_dca.calls == 1

    # 30프레임 윈도우가 아닌 프레임에는 보조 결과가 없다.
    for resp in responses[:29] + responses[30:]:
        assert "pentagon_scores" not in resp
        assert "dca" not in resp
        assert "rule_score" not in resp


def test_token_session_flow_finalizes_redis_session_on_disconnect(client, fake_redis, monkeypatch):
    """정상 종료 시 session:{id} status=finished + summary 저장 (DB 저장 없음)."""
    asyncio.run(seed_session_and_reference(fake_redis))
    monkeypatch.setattr(websocket_router, "get_dca_model_runner", lambda: None)

    with client.websocket_connect(f"/ws/analyze?token={make_token()}") as ws:
        ws.receive_json()
        _send_frames(ws, 31)

    status = asyncio.run(fake_redis.hget("session:test-session-1", "status"))
    assert status == b"finished"
    summary = json.loads(asyncio.run(fake_redis.get("session:test-session-1:summary")))
    assert summary["frame_count"] == 31
    assert summary["average_score"] > 0.99  # 프레임 점수 평균 (pentagon 값이 섞이지 않음)
    assert summary["pentagon_scores"]["window_count"] == 1


def test_token_flow_without_dca_checkpoint_keeps_rule_based_feedback(client, fake_redis, monkeypatch):
    """DCA checkpoint가 없으면 모델 로드만 실패하고 실시간 채점/pentagon은 계속 동작한다."""
    asyncio.run(seed_session_and_reference(fake_redis))
    monkeypatch.setattr(settings, "dca_checkpoint_path", "does/not/exist/best.pt")
    monkeypatch.setattr(dca_model_service, "_runner", None)
    monkeypatch.setattr(dca_model_service, "_attempted_key", None)

    with client.websocket_connect(f"/ws/analyze?token={make_token()}") as ws:
        ws.receive_json()
        responses = _send_frames(ws, 31)

    assert "pentagon_scores" in responses[29]
    assert all("dca" not in resp for resp in responses)
    assert all(resp["score"] > 0.99 for resp in responses)


def test_tracking_loss_frame_over_websocket_keeps_session_alive(client, fake_redis, monkeypatch):
    """골반이 추적 안 된(0,0,0) 프레임도 세션을 끊지 않고 유효 범위 점수를 돌려준다."""
    asyncio.run(seed_session_and_reference(fake_redis))
    monkeypatch.setattr(websocket_router, "get_dca_model_runner", lambda: None)

    frame = reference_frame(0)
    frame[23] = [0.0, 0.0, 0.0]
    frame[24] = [0.0, 0.0, 0.0]
    with client.websocket_connect(f"/ws/analyze?token={make_token()}") as ws:
        ws.receive_json()
        ws.send_json({"frame_idx": 0, "timestamp_ms": 0, "keypoints": frame})
        resp = ws.receive_json()
        assert resp["type"] == "feedback"
        assert -1.0 <= resp["score"] <= 1.0
        assert 23 not in resp["worst_joints"] and 24 not in resp["worst_joints"]


# ── 연결 방식 분리 (token 기본 / reference_id는 개발용 fallback) ─────────────


def test_missing_token_and_reference_closes_with_4001(client, fake_redis):
    with client.websocket_connect("/ws/analyze") as ws:
        assert "error" in ws.receive_json()
        with pytest.raises(Exception):
            ws.receive_json()


def test_reference_id_is_rejected_when_fallback_disabled(client, fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "enable_reference_id_fallback", False)
    with client.websocket_connect("/ws/analyze?reference_id=hollywood-action") as ws:
        assert "reference_id" in ws.receive_json()["error"]
        with pytest.raises(Exception):
            ws.receive_json()


def test_reference_id_fallback_works_when_enabled_and_skips_finalize(client, fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "enable_reference_id_fallback", True)
    monkeypatch.setattr(
        settings,
        "reference_keypoints",
        {"hollywood-action": {"video_id": 42, "keypoint_path": "not/a/local/file.npy"}},
    )
    monkeypatch.setattr(websocket_router, "get_dca_model_runner", lambda: None)
    asyncio.run(seed_session_and_reference(fake_redis))  # ref_kp:42 캐시 사용

    with client.websocket_connect("/ws/analyze?reference_id=hollywood-action") as ws:
        assert ws.receive_json() == {"status": "ready", "video_id": 42}
        resp = _send_frames(ws, 1)[0]
        assert resp["score"] > 0.99

    # fallback 연결은 BE 세션이 아니므로 session:{id}를 건드리지 않는다.
    assert asyncio.run(fake_redis.hget("session:test-session-1", "status")) == b"active"


def test_token_takes_priority_over_reference_id(client, fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "enable_reference_id_fallback", True)
    monkeypatch.setattr(websocket_router, "get_dca_model_runner", lambda: None)
    asyncio.run(seed_session_and_reference(fake_redis))

    with client.websocket_connect(
        f"/ws/analyze?token={make_token()}&reference_id=unknown-song"
    ) as ws:
        assert ws.receive_json() == {"status": "ready", "video_id": 42}


# ── 채점 통합: 시간 정렬(integration) + PR #7 채점 ────────────────────────


def test_time_alignment_picks_delayed_frame_with_realistic_reference():
    """integration의 시간 정렬 의도를 실제 관절 분포가 있는 reference로 검증."""
    reference = make_reference_sequence()
    expected_idx, performed_idx = 10, 7  # 사용자가 3프레임 늦게 따라하는 상황
    result = compute_feedback(
        reference,
        reference[performed_idx],
        frame_idx=0,
        timestamp_ms=round(expected_idx * 1000 / 30),
        fps=30.0,
    )
    assert result["frame_idx"] == performed_idx
    assert result["score"] > 0.99


def test_translation_and_scale_invariance_with_realistic_reference():
    """integration의 카메라 위치/거리 불변 의도를 PR #7 채점으로도 만족하는지 검증."""
    reference = make_reference_sequence()
    incoming = reference[5] * 1.8 + np.array([4.0, -3.0, 0.5], dtype=np.float32)
    result = compute_feedback(reference, incoming, frame_idx=5)
    assert result["score"] > 0.99


def test_time_alignment_does_not_hide_a_wrong_pose():
    """정렬 범위 안 어느 프레임과도 다른 자세면 여전히 낮은 점수여야 한다."""
    reference = make_reference_sequence()
    wrong = reference[5].copy()
    wrong[[15, 16, 27, 28]] = wrong[[16, 15, 28, 27]] + np.array([0.3, -0.3, 0.0], dtype=np.float32)
    result = compute_feedback(reference, wrong, frame_idx=5)
    assert result["score"] < settings.feedback_threshold_good


# ── DCA 응답 계약 (checkpoint 없이, 학습되지 않은 모델 구조로만) ─────────────


def test_dca_runner_output_contains_fields_fe_uses():
    """checkpoint 없이 모델 구조만 메모리에 만들어 predict 응답 형태를 확인한다.

    학습된 가중치가 아니므로 점수 값 자체는 의미가 없다 — FE가 쓰는
    frontend_payload.score_100 / highlight_joints 필드와 형태만 검증한다.
    """
    from model import create_dca_model

    runner = dca_model_service.DcaModelRunner("unused-no-checkpoint", device="cpu")
    runner.device = torch.device("cpu")
    runner.model = create_dca_model().to(runner.device)

    reference = make_reference_sequence()[:30]
    result = runner.predict(reference, reference)
    payload = result["frontend_payload"]
    assert 0.0 <= payload["score_100"] <= 100.0
    assert isinstance(payload["highlight_joints"], list)
    assert all(0 <= j < 33 for j in payload["highlight_joints"])
