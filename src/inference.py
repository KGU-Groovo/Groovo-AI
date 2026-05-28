from pathlib import Path
import json

import numpy as np
import torch


def to_jsonable(value):
    """
    torch / numpy 타입을 JSON 저장 가능한 Python 기본 타입으로 변환한다.
    """
    if torch.is_tensor(value):
        return to_jsonable(value.detach().cpu().numpy())

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]

    return value


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)


def load_json(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_model(model, checkpoint_path, device):
    """
    학습된 best checkpoint를 모델에 로드한다.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint를 찾을 수 없습니다: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return checkpoint


def ensure_batch_sequence(seq, device):
    """
    입력 sequence를 모델 입력 형태로 맞춘다.

    허용 shape:
    - (30, 33, 3)
    - (B, 30, 33, 3)

    반환:
    - torch.float32 tensor
    - shape (B, 30, 33, 3)
    """
    if isinstance(seq, np.ndarray):
        seq = torch.from_numpy(seq)

    if not torch.is_tensor(seq):
        raise ValueError("seq는 np.ndarray 또는 torch.Tensor여야 합니다.")

    seq = seq.to(dtype=torch.float32)

    if seq.ndim == 3:
        if tuple(seq.shape) != (30, 33, 3):
            raise ValueError(
                f"3차원 입력 shape는 (30, 33, 3)이어야 합니다. 현재: {tuple(seq.shape)}"
            )
        seq = seq.unsqueeze(0)

    elif seq.ndim == 4:
        if tuple(seq.shape[1:]) != (30, 33, 3):
            raise ValueError(
                f"4차원 입력 shape는 (B, 30, 33, 3)이어야 합니다. 현재: {tuple(seq.shape)}"
            )

    else:
        raise ValueError(f"입력은 3차원 또는 4차원이어야 합니다. 현재 ndim: {seq.ndim}")

    return seq.to(device)


def score_to_color(score_100, green_threshold=80.0, orange_threshold=60.0):
    """
    점수를 화면 색상으로 변환한다.

    green  : 잘 맞음
    orange : 보통
    red    : 많이 틀림
    """
    score_100 = float(score_100)

    if score_100 >= green_threshold:
        return "green"

    if score_100 >= orange_threshold:
        return "orange"

    return "red"


def color_to_status(color):
    if color == "green":
        return "good"

    if color == "orange":
        return "normal"

    return "bad"


def score_to_status(score_100, green_threshold=80.0, orange_threshold=60.0):
    color = score_to_color(
        score_100,
        green_threshold=green_threshold,
        orange_threshold=orange_threshold,
    )
    return color_to_status(color)


def make_frontend_payload(
    score_100,
    color,
    status,
    highlight_joints,
    pred_joint_errors,
    top_k=3,
):
    """
    프론트엔드/서버에 넘기기 쉬운 dict를 만든다.
    """
    score_100 = float(np.clip(score_100, 0.0, 100.0))
    highlight_joints = [int(x) for x in highlight_joints]
    pred_joint_errors = [float(x) for x in pred_joint_errors]

    joint_errors = [
        {
            "joint_index": int(joint_index),
            "error": float(error),
        }
        for joint_index, error in enumerate(pred_joint_errors)
    ]

    top_indices = np.argsort(np.asarray(pred_joint_errors))[::-1][:top_k]

    top_joint_errors = [
        {
            "joint_index": int(idx),
            "error": float(pred_joint_errors[idx]),
        }
        for idx in top_indices
    ]

    return {
        "score_100": round(score_100, 4),
        "color": color,
        "status": status,
        "highlight_joints": highlight_joints,
        "joint_errors": joint_errors,
        "top_joint_errors": top_joint_errors,
    }


def run_single_inference(
    model,
    idol_seq,
    user_seq,
    device,
    calibration_offset=0.0,
    green_threshold=80.0,
    orange_threshold=60.0,
    top_k=3,
):
    """
    sample 1개에 대한 추론 함수.

    현재 모델은 rule-based pseudo label 기반 prototype이다.
    실제 인간 평가 점수를 직접 학습한 모델은 아니다.
    """
    idol_seq = ensure_batch_sequence(idol_seq, device)
    user_seq = ensure_batch_sequence(user_seq, device)

    if idol_seq.shape[0] != 1 or user_seq.shape[0] != 1:
        raise ValueError("run_single_inference는 batch size 1 입력만 받습니다.")

    if idol_seq.shape != user_seq.shape:
        raise ValueError(
            f"idol_seq와 user_seq shape가 같아야 합니다. "
            f"idol={tuple(idol_seq.shape)}, user={tuple(user_seq.shape)}"
        )

    model.eval()

    with torch.no_grad():
        outputs = model(idol_seq, user_seq)

    raw_score_100 = float(outputs["score_100"][0].detach().cpu().item())
    raw_score_norm = float(outputs["score_norm"][0].detach().cpu().item())

    # calibration_offset은 기본 0.0이다.
    # 나중에 모델이 평균적으로 후하게 예측하면 -5~-8 정도로 조정 가능하다.
    calibrated_score_100 = float(np.clip(raw_score_100 + calibration_offset, 0.0, 100.0))
    calibrated_score_norm = float(calibrated_score_100 / 100.0)

    color = score_to_color(
        calibrated_score_100,
        green_threshold=green_threshold,
        orange_threshold=orange_threshold,
    )

    status = color_to_status(color)

    highlight_joints = (
        outputs["highlight_joints"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(int)
        .tolist()
    )

    pred_joint_errors = (
        outputs["pred_joint_errors"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(float)
        .tolist()
    )

    frontend_payload = make_frontend_payload(
        calibrated_score_100,
        color,
        status,
        highlight_joints,
        pred_joint_errors,
        top_k=top_k,
    )

    return {
        "raw_score_100": raw_score_100,
        "raw_score_norm": raw_score_norm,
        "calibrated_score_100": calibrated_score_100,
        "score_norm": calibrated_score_norm,
        "color": color,
        "status": status,
        "highlight_joints": highlight_joints,
        "pred_joint_errors": pred_joint_errors,
        "frontend_payload": frontend_payload,
        "note": "Prototype inference based on rule-based pseudo labels.",
    }


def run_batch_inference(
    model,
    idol_batch,
    user_batch,
    device,
    calibration_offset=0.0,
    green_threshold=80.0,
    orange_threshold=60.0,
    top_k=3,
):
    """
    batch 단위 추론 함수.

    입력:
    - idol_batch: (B, 30, 33, 3)
    - user_batch: (B, 30, 33, 3)
    """
    idol_batch = ensure_batch_sequence(idol_batch, device)
    user_batch = ensure_batch_sequence(user_batch, device)

    if idol_batch.shape != user_batch.shape:
        raise ValueError(
            f"idol_batch와 user_batch shape가 같아야 합니다. "
            f"idol={tuple(idol_batch.shape)}, user={tuple(user_batch.shape)}"
        )

    model.eval()

    with torch.no_grad():
        outputs = model(idol_batch, user_batch)

    raw_scores_100 = outputs["score_100"].detach().cpu().numpy()
    raw_scores_norm = outputs["score_norm"].detach().cpu().numpy()
    highlight_joints_batch = outputs["highlight_joints"].detach().cpu().numpy()
    pred_joint_errors_batch = outputs["pred_joint_errors"].detach().cpu().numpy()

    results = []

    batch_size = int(idol_batch.shape[0])

    for i in range(batch_size):
        raw_score_100 = float(raw_scores_100[i])
        raw_score_norm = float(raw_scores_norm[i])

        calibrated_score_100 = float(
            np.clip(raw_score_100 + calibration_offset, 0.0, 100.0)
        )

        calibrated_score_norm = float(calibrated_score_100 / 100.0)

        color = score_to_color(
            calibrated_score_100,
            green_threshold=green_threshold,
            orange_threshold=orange_threshold,
        )

        status = color_to_status(color)

        highlight_joints = highlight_joints_batch[i].astype(int).tolist()
        pred_joint_errors = pred_joint_errors_batch[i].astype(float).tolist()

        frontend_payload = make_frontend_payload(
            calibrated_score_100,
            color,
            status,
            highlight_joints,
            pred_joint_errors,
            top_k=top_k,
        )

        results.append(
            {
                "sample_order": int(i),
                "raw_score_100": raw_score_100,
                "raw_score_norm": raw_score_norm,
                "calibrated_score_100": calibrated_score_100,
                "score_norm": calibrated_score_norm,
                "color": color,
                "status": status,
                "highlight_joints": highlight_joints,
                "pred_joint_errors": pred_joint_errors,
                "frontend_payload": frontend_payload,
                "note": "Prototype inference based on rule-based pseudo labels.",
            }
        )

    return results


def summarize_inference_result(result):
    print("===== Inference Result =====")
    print("raw_score_100:", result["raw_score_100"])
    print("calibrated_score_100:", result["calibrated_score_100"])
    print("score_norm:", result["score_norm"])
    print("color:", result["color"])
    print("status:", result["status"])
    print("highlight_joints:", result["highlight_joints"])
    print("note:", result["note"])
