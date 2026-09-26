import sys
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

sys.modules.setdefault("aioboto3", SimpleNamespace(Session=object))

from app.config import settings
from app.main import app
import app.routers.websocket as websocket_router


def test_analyze_websocket_accepts_an_allowed_reference_without_a_token(monkeypatch):
    async def load_reference_keypoints(_redis, _video_id, _keypoint_path):
        return np.ones((2, 33, 3), dtype=np.float32)

    monkeypatch.setattr(
        settings,
        "reference_keypoints",
        {"hollywood-action": {"video_id": 1, "keypoint_path": "refs/hollywood.npy"}},
        raising=False,
    )
    monkeypatch.setattr(websocket_router, "load_reference_keypoints", load_reference_keypoints)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/analyze?reference_id=hollywood-action") as socket:
            assert socket.receive_json() == {"status": "ready", "video_id": 1}
            socket.send_json({"keypoints": np.ones((33, 3)).tolist(), "frame_idx": 0})
            assert socket.receive_json()["score"] == 1.0


def test_analyze_websocket_returns_pentagon_scores_after_30_valid_frames(monkeypatch):
    async def load_reference_keypoints(_redis, _video_id, _keypoint_path):
        return np.ones((60, 33, 3), dtype=np.float32)

    monkeypatch.setattr(
        settings,
        "reference_keypoints",
        {"hollywood-action": {"video_id": 1, "keypoint_path": "refs/hollywood.npy"}},
    )
    monkeypatch.setattr(websocket_router, "load_reference_keypoints", load_reference_keypoints)
    monkeypatch.setattr(
        websocket_router,
        "score_window",
        lambda _user, _reference: {"final_score": 91.2, "scores": {"accuracy": 90.0}},
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws/analyze?reference_id=hollywood-action") as socket:
            socket.receive_json()
            for frame_idx in range(30):
                socket.send_json({"keypoints": np.ones((33, 3)).tolist(), "frame_idx": frame_idx})
                feedback = socket.receive_json()

    assert feedback["pentagon_scores"] == {"final_score": 90.0, "scores": {"accuracy": 90.0}}
    assert feedback["score"] == 0.9
    assert feedback["pentagon_scores"]["final_score"] == 90.0


def test_analyze_websocket_returns_aggregated_summary_when_session_completes(monkeypatch):
    async def load_reference_keypoints(_redis, _video_id, _keypoint_path):
        return np.ones((90, 33, 3), dtype=np.float32)

    scores = iter([80.0, 100.0])

    def score_window(_user, _reference):
        score = next(scores)
        return {
            "final_score": score,
            "scores": {
                "timing": score,
                "balance": score,
                "rhythm": score,
                "detail": score,
                "accuracy": score,
            },
        }

    monkeypatch.setattr(
        settings,
        "reference_keypoints",
        {"hollywood-action": {"video_id": 1, "keypoint_path": "refs/hollywood.npy"}},
    )
    monkeypatch.setattr(websocket_router, "load_reference_keypoints", load_reference_keypoints)
    monkeypatch.setattr(websocket_router, "score_window", score_window)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/analyze?reference_id=hollywood-action") as socket:
            socket.receive_json()
            for frame_idx in range(60):
                socket.send_json({"keypoints": np.ones((33, 3)).tolist(), "frame_idx": frame_idx})
                socket.receive_json()
            socket.send_json({"type": "complete"})
            summary = socket.receive_json()

    assert summary == {
        "type": "session_summary",
        "session_summary": {
            "final_score": 90.0,
            "scores": {
                "timing": 90.0,
                "balance": 90.0,
                "rhythm": 90.0,
                "detail": 90.0,
                "accuracy": 90.0,
            },
            "window_count": 2,
        },
    }


def test_analyze_websocket_discards_partial_window_while_body_is_not_visible(monkeypatch):
    async def load_reference_keypoints(_redis, _video_id, _keypoint_path):
        return np.ones((60, 33, 3), dtype=np.float32)

    calls = 0

    def score_window(_user, _reference):
        nonlocal calls
        calls += 1
        return {"final_score": 91.2, "scores": {"accuracy": 90.0}}

    monkeypatch.setattr(
        settings,
        "reference_keypoints",
        {"hollywood-action": {"video_id": 1, "keypoint_path": "refs/hollywood.npy"}},
    )
    monkeypatch.setattr(websocket_router, "load_reference_keypoints", load_reference_keypoints)
    monkeypatch.setattr(websocket_router, "score_window", score_window)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/analyze?reference_id=hollywood-action") as socket:
            socket.receive_json()
            for frame_idx in range(29):
                socket.send_json({"keypoints": np.ones((33, 3)).tolist(), "frame_idx": frame_idx})
                socket.receive_json()
            socket.send_json({"body_visible": False})
            socket.send_json({"keypoints": np.ones((33, 3)).tolist(), "frame_idx": 29})
            socket.receive_json()

    assert calls == 0


def test_analyze_websocket_uses_dca_score_after_a_full_window(monkeypatch):
    class FakeDcaModel:
        def predict(self, reference_window, user_window):
            assert reference_window.shape == (30, 33, 3)
            assert user_window.shape == (30, 33, 3)
            return {
                "score_norm": 0.87,
                "frontend_payload": {"score_100": 87.0, "highlight_joints": [11, 13, 15]},
            }

    async def load_reference_keypoints(_redis, _video_id, _keypoint_path):
        return np.ones((60, 33, 3), dtype=np.float32)

    monkeypatch.setattr(
        settings,
        "reference_keypoints",
        {"hollywood-action": {"video_id": 1, "keypoint_path": "refs/hollywood.npy"}},
    )
    monkeypatch.setattr(websocket_router, "load_reference_keypoints", load_reference_keypoints)
    monkeypatch.setattr(websocket_router, "get_dca_model_runner", lambda: FakeDcaModel())

    with TestClient(app) as client:
        with client.websocket_connect("/ws/analyze?reference_id=hollywood-action") as socket:
            socket.receive_json()
            for frame_idx in range(30):
                socket.send_json({"keypoints": np.ones((33, 3)).tolist(), "frame_idx": frame_idx})
                feedback = socket.receive_json()

    assert feedback["score"] == 1.0
    assert feedback["rule_score"] == 1.0
    assert feedback["dca"] == {"score_100": 87.0, "highlight_joints": [11, 13, 15]}
