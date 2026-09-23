import logging
import sys
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


class DcaModelRunner:
    """하나의 읽기 전용 DCA 모델로 독립적인 WebSocket 창을 추론한다."""

    def __init__(self, checkpoint_path, device="auto", normalize_input=False):
        self.checkpoint_path = Path(checkpoint_path)
        self.device_name = device
        self.normalize_input = normalize_input
        self.model = None
        self.device = None
        self._predict_lock = threading.Lock()

    def load(self):
        import torch
        from inference import load_checkpoint_model
        from model import create_dca_model

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"DCA checkpoint를 찾을 수 없습니다: {self.checkpoint_path}")
        if self.device_name == "auto":
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(self.device_name)
        self.model = create_dca_model().to(self.device)
        load_checkpoint_model(self.model, self.checkpoint_path, self.device)
        return self

    def predict(self, reference_window, user_window):
        from inference import run_single_inference

        if self.model is None:
            raise RuntimeError("DCA model이 로드되지 않았습니다.")
        with self._predict_lock:
            return run_single_inference(
                self.model,
                np.asarray(reference_window, dtype=np.float32),
                np.asarray(user_window, dtype=np.float32),
                self.device,
                normalize_input=self.normalize_input,
            )


_runner: DcaModelRunner | None = None
_attempted_key: tuple[str, str, bool] | None = None


def get_dca_model_runner() -> DcaModelRunner | None:
    """설정된 checkpoint를 한 번만 로드하고, 실패 시 규칙 기반 점수를 유지한다."""
    global _runner, _attempted_key

    from app.config import settings

    key = (settings.dca_checkpoint_path, settings.dca_device, settings.dca_normalize_input)
    if _runner is not None and _attempted_key == key:
        return _runner
    if _attempted_key == key:
        return None

    _attempted_key = key
    try:
        _runner = DcaModelRunner(*key).load()
        logger.info("DCA model loaded: %s", key[0])
    except Exception:
        _runner = None
        logger.exception("DCA model load failed; rule-based feedback remains active")
    return _runner
