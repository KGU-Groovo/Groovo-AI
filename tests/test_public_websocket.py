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
