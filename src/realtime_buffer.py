from collections import deque

import numpy as np
import torch

from inference import run_single_inference


def validate_pose_frame(frame, num_joints=33, coord_dim=3):
    """
    실시간으로 들어오는 pose frame 1개를 검사한다.

    기대 shape:
    - (33, 3)
    - 33개 관절
    - 각 관절은 x, y, z 좌표 3개
    """
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()

    if not isinstance(frame, np.ndarray):
        raise ValueError("frame은 np.ndarray 또는 torch.Tensor여야 합니다.")

    if frame.shape != (num_joints, coord_dim):
        raise ValueError(
            f"frame shape는 ({num_joints}, {coord_dim})이어야 합니다. "
            f"현재 shape: {frame.shape}"
        )

    frame = frame.astype(np.float32)

    if np.isnan(frame).any():
        raise ValueError("frame에 NaN 값이 있습니다.")

    if np.isinf(frame).any():
        raise ValueError("frame에 Inf 값이 있습니다.")

    return frame


class SlidingWindowPoseBuffer:
    """
    실시간 pose frame을 최근 30프레임 단위로 관리하는 buffer.

    서버에서는 add_frame_and_predict(frame)을 호출하면 된다.
    - 30프레임 미만이면 ready=False
    - 30프레임 이상이면 DCA-Net v2 추론 결과 반환
    """

    def __init__(
        self,
        model,
        idol_seq,
        device,
        window_size=30,
        num_joints=33,
        coord_dim=3,
        calibration_offset=0.0,
        green_threshold=80.0,
        orange_threshold=60.0,
        top_k=3,
    ):
        self.model = model
        self.device = device
        self.window_size = int(window_size)
        self.num_joints = int(num_joints)
        self.coord_dim = int(coord_dim)
        self.calibration_offset = float(calibration_offset)
        self.green_threshold = float(green_threshold)
        self.orange_threshold = float(orange_threshold)
        self.top_k = int(top_k)

        self.idol_seq = self._prepare_idol_seq(idol_seq)

        self.user_buffer = deque(maxlen=self.window_size)
        self.frame_count = 0
        self.last_result = None

    def _prepare_idol_seq(self, idol_seq):
        if isinstance(idol_seq, torch.Tensor):
            idol_seq = idol_seq.detach().cpu().numpy()

        if not isinstance(idol_seq, np.ndarray):
            raise ValueError("idol_seq는 np.ndarray 또는 torch.Tensor여야 합니다.")

        idol_seq = idol_seq.astype(np.float32)

        if idol_seq.shape == (1, self.window_size, self.num_joints, self.coord_dim):
            idol_seq = idol_seq[0]

        expected_shape = (self.window_size, self.num_joints, self.coord_dim)

        if idol_seq.shape != expected_shape:
            raise ValueError(
                f"idol_seq shape는 {expected_shape} 또는 "
                f"(1, {self.window_size}, {self.num_joints}, {self.coord_dim})이어야 합니다. "
                f"현재 shape: {idol_seq.shape}"
            )

        if np.isnan(idol_seq).any():
            raise ValueError("idol_seq에 NaN 값이 있습니다.")

        if np.isinf(idol_seq).any():
            raise ValueError("idol_seq에 Inf 값이 있습니다.")

        return idol_seq

    def reset(self):
        self.user_buffer.clear()
        self.frame_count = 0
        self.last_result = None

    def add_user_frame(self, frame):
        frame = validate_pose_frame(
            frame,
            num_joints=self.num_joints,
            coord_dim=self.coord_dim,
        )

        self.user_buffer.append(frame)
        self.frame_count += 1

        return {
            "frame_count": int(self.frame_count),
            "buffer_size": int(len(self.user_buffer)),
            "ready": bool(self.is_ready()),
        }

    def is_ready(self):
        return len(self.user_buffer) == self.window_size

    def get_user_sequence(self):
        if not self.is_ready():
            raise ValueError(
                f"아직 {self.window_size}프레임이 모이지 않았습니다. "
                f"현재 buffer_size: {len(self.user_buffer)}"
            )

        return np.stack(list(self.user_buffer), axis=0).astype(np.float32)

    def predict(self):
        if not self.is_ready():
            return {
                "ready": False,
                "message": f"Need {self.window_size} frames before inference.",
                "frame_count": int(self.frame_count),
                "buffer_size": int(len(self.user_buffer)),
                "window_size": int(self.window_size),
            }

        user_seq = self.get_user_sequence()

        result = run_single_inference(
            self.model,
            self.idol_seq,
            user_seq,
            self.device,
            calibration_offset=self.calibration_offset,
            green_threshold=self.green_threshold,
            orange_threshold=self.orange_threshold,
            top_k=self.top_k,
        )

        result["ready"] = True
        result["frame_count"] = int(self.frame_count)
        result["buffer_size"] = int(len(self.user_buffer))
        result["window_size"] = int(self.window_size)

        self.last_result = result

        return result

    def add_frame_and_predict(self, frame):
        state = self.add_user_frame(frame)

        if not state["ready"]:
            return {
                "ready": False,
                "message": f"Need {self.window_size} frames before inference.",
                "frame_count": int(state["frame_count"]),
                "buffer_size": int(state["buffer_size"]),
                "window_size": int(self.window_size),
            }

        return self.predict()

    def get_state(self):
        return {
            "frame_count": int(self.frame_count),
            "buffer_size": int(len(self.user_buffer)),
            "window_size": int(self.window_size),
            "ready": bool(self.is_ready()),
            "has_last_result": bool(self.last_result is not None),
        }


def simulate_realtime_sequence(buffer, user_seq, predict_every=1, reset_buffer=False):
    """
    user_seq를 frame-by-frame으로 넣어 실시간 입력을 흉내 낸다.

    user_seq shape:
    - (30, 33, 3)
    - 또는 더 긴 (T, 33, 3)

    predict_every:
    - 1이면 ready 이후 매 프레임 추론
    - 2이면 ready 이후 2프레임마다 추론
    """
    if reset_buffer:
        buffer.reset()

    if isinstance(user_seq, torch.Tensor):
        user_seq = user_seq.detach().cpu().numpy()

    if not isinstance(user_seq, np.ndarray):
        raise ValueError("user_seq는 np.ndarray 또는 torch.Tensor여야 합니다.")

    if user_seq.ndim != 3:
        raise ValueError(
            f"user_seq는 (T, 33, 3) shape여야 합니다. 현재 ndim: {user_seq.ndim}"
        )

    if user_seq.shape[1:] != (buffer.num_joints, buffer.coord_dim):
        raise ValueError(
            f"user_seq shape는 (T, {buffer.num_joints}, {buffer.coord_dim})이어야 합니다. "
            f"현재 shape: {user_seq.shape}"
        )

    if predict_every < 1:
        raise ValueError("predict_every는 1 이상이어야 합니다.")

    all_steps = []
    ready_results = []

    for frame_idx in range(user_seq.shape[0]):
        state = buffer.add_user_frame(user_seq[frame_idx])

        should_predict = (
            state["ready"]
            and ((frame_idx + 1 - buffer.window_size) % predict_every == 0)
        )

        if should_predict:
            result = buffer.predict()
        else:
            result = {
                "ready": False,
                "message": "Frame added. Prediction skipped or buffer not ready.",
                "frame_count": int(state["frame_count"]),
                "buffer_size": int(state["buffer_size"]),
                "window_size": int(buffer.window_size),
            }

        all_steps.append(result)

        if result.get("ready") is True:
            ready_results.append(result)

    return {
        "num_input_frames": int(user_seq.shape[0]),
        "num_predictions": int(len(ready_results)),
        "all_steps": all_steps,
        "ready_results": ready_results,
    }
