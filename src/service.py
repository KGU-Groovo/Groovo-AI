from pathlib import Path

import numpy as np
import torch

from model import create_dca_model, count_parameters
from inference import (
    load_checkpoint_model,
    run_single_inference,
)
from realtime_buffer import SlidingWindowPoseBuffer, validate_pose_frame


class GroovoDCAService:
    """
    서버팀이 사용할 고수준 wrapper.

    역할:
    - 모델 로드
    - 기준 안무 설정
    - 30프레임 sequence 추론
    - 실시간 frame-by-frame 추론

    주의:
    - 이 파일에는 학습 코드가 없다.
    - 현재 모델은 rule-based pseudo label로 학습된 prototype이다.
    """

    def __init__(
        self,
        checkpoint_path,
        device=None,
        calibration_offset=0.0,
        green_threshold=80.0,
        orange_threshold=60.0,
        top_k=3,
        window_size=30,
    ):
        self.checkpoint_path = Path(checkpoint_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        self.calibration_offset = float(calibration_offset)
        self.green_threshold = float(green_threshold)
        self.orange_threshold = float(orange_threshold)
        self.top_k = int(top_k)
        self.window_size = int(window_size)

        self.model = None
        self.checkpoint = None
        self.reference_idol_seq = None
        self.realtime_buffer = None

        self.total_params = None
        self.trainable_params = None

    def load_model(self):
        """
        checkpoint에서 DCA-Net v2 모델을 로드한다.
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint를 찾을 수 없습니다: {self.checkpoint_path}")

        self.model = create_dca_model().to(self.device)

        self.checkpoint = load_checkpoint_model(
            self.model,
            self.checkpoint_path,
            self.device,
        )

        self.model.eval()

        self.total_params, self.trainable_params = count_parameters(self.model)

        return self

    def is_model_loaded(self):
        return self.model is not None

    def _prepare_reference(self, idol_seq):
        """
        기준 안무 sequence shape 검증.

        허용:
        - (30, 33, 3)
        - (1, 30, 33, 3)
        """
        if isinstance(idol_seq, torch.Tensor):
            idol_seq = idol_seq.detach().cpu().numpy()

        if not isinstance(idol_seq, np.ndarray):
            raise ValueError("idol_seq는 np.ndarray 또는 torch.Tensor여야 합니다.")

        idol_seq = idol_seq.astype(np.float32)

        if idol_seq.shape == (1, self.window_size, 33, 3):
            idol_seq = idol_seq[0]

        expected_shape = (self.window_size, 33, 3)

        if idol_seq.shape != expected_shape:
            raise ValueError(
                f"idol_seq shape는 {expected_shape} 또는 "
                f"(1, {self.window_size}, 33, 3)이어야 합니다. "
                f"현재 shape: {idol_seq.shape}"
            )

        if np.isnan(idol_seq).any():
            raise ValueError("idol_seq에 NaN 값이 있습니다.")

        if np.isinf(idol_seq).any():
            raise ValueError("idol_seq에 Inf 값이 있습니다.")

        return idol_seq

    def set_reference(self, idol_seq):
        """
        기준 안무 sequence를 설정하고 realtime buffer를 초기화한다.
        """
        if not self.is_model_loaded():
            raise ValueError("먼저 load_model()을 호출해야 합니다.")

        self.reference_idol_seq = self._prepare_reference(idol_seq)

        self.realtime_buffer = SlidingWindowPoseBuffer(
            model=self.model,
            idol_seq=self.reference_idol_seq,
            device=self.device,
            window_size=self.window_size,
            calibration_offset=self.calibration_offset,
            green_threshold=self.green_threshold,
            orange_threshold=self.orange_threshold,
            top_k=self.top_k,
        )

        return self.get_state()

    def predict_sequence(self, user_seq):
        """
        이미 30프레임 user_seq가 준비되어 있을 때 사용하는 직접 추론 함수.
        """
        if not self.is_model_loaded():
            raise ValueError("먼저 load_model()을 호출해야 합니다.")

        if self.reference_idol_seq is None:
            raise ValueError("먼저 set_reference(idol_seq)를 호출해야 합니다.")

        result = run_single_inference(
            self.model,
            self.reference_idol_seq,
            user_seq,
            self.device,
            calibration_offset=self.calibration_offset,
            green_threshold=self.green_threshold,
            orange_threshold=self.orange_threshold,
            top_k=self.top_k,
        )

        return result

    def add_frame_and_predict(self, user_frame):
        """
        서버 실시간 입력용 메인 함수.

        user_frame shape:
        - (33, 3)

        반환:
        - 30프레임 전: ready=False
        - 30프레임 이후: score/color/highlight_joints 포함 결과
        """
        if not self.is_model_loaded():
            raise ValueError("먼저 load_model()을 호출해야 합니다.")

        if self.realtime_buffer is None:
            raise ValueError("먼저 set_reference(idol_seq)를 호출해야 합니다.")

        validate_pose_frame(user_frame)
        return self.realtime_buffer.add_frame_and_predict(user_frame)

    def reset_buffer(self):
        if self.realtime_buffer is None:
            raise ValueError("먼저 set_reference(idol_seq)를 호출해야 합니다.")

        self.realtime_buffer.reset()
        return self.get_state()

    def get_state(self):
        if self.realtime_buffer is None:
            buffer_state = None
        else:
            buffer_state = self.realtime_buffer.get_state()

        if self.checkpoint is None:
            checkpoint_epoch = None
            best_valid_loss = None
        else:
            checkpoint_epoch = self.checkpoint.get("epoch")
            best_valid_loss = self.checkpoint.get("best_valid_loss")

        return {
            "model_loaded": bool(self.is_model_loaded()),
            "has_reference": bool(self.reference_idol_seq is not None),
            "device": str(self.device),
            "checkpoint_epoch": checkpoint_epoch,
            "best_valid_loss": best_valid_loss,
            "num_parameters": self.total_params,
            "num_trainable_parameters": self.trainable_params,
            "buffer_state": buffer_state,
        }

    def get_model_info(self):
        return {
            "model_name": "Groovo DCA-Net v2 Prototype",
            "model_version": "dca_v2_feature_diff_injection",
            "frame_length": self.window_size,
            "num_joints": 33,
            "coord_dim": 3,
            "input_shapes": {
                "idol_seq": [self.window_size, 33, 3],
                "user_seq": [self.window_size, 33, 3],
                "single_frame": [33, 3],
            },
            "output_fields": [
                "ready",
                "score_100",
                "color",
                "status",
                "highlight_joints",
                "pred_joint_errors",
                "frontend_payload",
            ],
            "color_rule": {
                "green": f"score_100 >= {self.green_threshold}",
                "orange": f"{self.orange_threshold} <= score_100 < {self.green_threshold}",
                "red": f"score_100 < {self.orange_threshold}",
            },
            "realtime_rule": {
                "window_size": self.window_size,
                "first_ready_frame": self.window_size,
                "before_ready": "ready=False",
            },
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "num_parameters": self.total_params,
            "num_trainable_parameters": self.trainable_params,
            "limitations": [
                "This model is trained on rule-based pseudo labels, not real human scores.",
                "Scores should be treated as prototype feedback.",
                "Service performance must be revalidated with real user dance data.",
                "Color thresholds may need calibration for production.",
            ],
        }


def make_server_input_example():
    return {
        "mode": "realtime",
        "reference_id": "exo_love_shot_f_view",
        "timestamp_ms": 123456,
        "incoming_keypoints_shape": [33, 3],
        "expected_frame_shape": [33, 3],
        "sequence_shape_for_direct_prediction": [30, 33, 3],
        "note": "Send one MediaPipe Pose frame per request for realtime mode, or send 30 frames for sequence mode.",
    }


def make_server_output_example(result):
    return {
        "ready": bool(result.get("ready", True)),
        "score_100": result.get("calibrated_score_100", result.get("score_100")),
        "color": result.get("color"),
        "status": result.get("status"),
        "highlight_joints": result.get("highlight_joints"),
        "frontend_payload": result.get("frontend_payload"),
        "note": "Prototype output based on rule-based pseudo labels.",
    }


def make_model_spec_markdown(model_info):
    """
    Notion/GitHub README에 붙일 수 있는 모델 스펙 문서 생성.
    """
    lines = [
        "# Groovo DCA-Net v2 Model Spec",
        "",
        "## Purpose",
        "",
        "Groovo DCA-Net v2 compares a reference idol dance sequence with a user dance sequence and returns a prototype dance similarity score.",
        "",
        "## Important Limitation",
        "",
        "This model was trained on rule-based pseudo labels, not real human scoring labels. Treat outputs as prototype feedback.",
        "",
        "## Input Shape",
        "",
        "- `idol_seq`: `(30, 33, 3)`",
        "- `user_seq`: `(30, 33, 3)`",
        "- `single_frame`: `(33, 3)` for realtime buffer input",
        "",
        "Each frame uses MediaPipe Pose 33 joints with x, y, z coordinates.",
        "",
        "## Output Format",
        "",
        "- `score_100`: dance similarity score from 0 to 100",
        "- `color`: `red`, `orange`, or `green`",
        "- `status`: `bad`, `normal`, or `good`",
        "- `highlight_joints`: top joints with larger predicted errors",
        "- `frontend_payload`: server/frontend friendly response dict",
        "",
        "## Color Rule",
        "",
        "- green: `score_100 >= 80`",
        "- orange: `60 <= score_100 < 80`",
        "- red: `score_100 < 60`",
        "",
        "## Realtime Buffer",
        "",
        "The model requires 30 frames. The realtime wrapper stores recent user pose frames.",
        "",
        "- Frames 1-29: `ready=False`",
        "- Frame 30 and later: inference result is returned",
        "",
        "## Current Performance Summary",
        "",
        "- Final test MAE_100: about 11 points",
        "- Final correlation: about 0.9",
        "- Leakage check: `leakage_detected=False`",
        "",
        "## Server Usage Example",
        "",
        "```python",
        "from service import GroovoDCAService",
        "",
        "service = GroovoDCAService('/path/to/dca_v2_best_model.pt')",
        "service.load_model()",
        "service.set_reference(idol_seq)",
        "",
        "# realtime mode",
        "result = service.add_frame_and_predict(user_frame)",
        "",
        "# sequence mode",
        "result = service.predict_sequence(user_seq)",
        "```",
        "",
        "## Model Info",
        "",
        f"- model_name: {model_info.get('model_name')}",
        f"- model_version: {model_info.get('model_version')}",
        f"- checkpoint_path: {model_info.get('checkpoint_path')}",
        f"- num_parameters: {model_info.get('num_parameters')}",
        "",
        "## Limitations",
        "",
    ]

    for item in model_info.get("limitations", []):
        lines.append(f"- {item}")

    return "\n".join(lines)
