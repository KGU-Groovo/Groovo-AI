import asyncio
import json
import logging
from collections import deque

import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.config import settings
from app.redis_client import get_redis
from app.services.keypoint_service import (
    KEYPOINT_DIM,
    NUM_LANDMARKS,
    compute_feedback,
    load_reference_keypoints,
)
from app.services.pentagon_scoring_service import (
    WINDOW_SIZE,
    score_window,
    window_has_reference_seam,
)

router = APIRouter()
logger = logging.getLogger(__name__)

_WARN_SEC = 3.0    # 경고 임계값: 3초 동안 수신 없으면 마지막 점수 유지 + 경고
_PAUSE_SEC = 10.0  # 일시정지 권고 임계값: 10초


@router.websocket("/ws/analyze")
async def analyze_websocket(
    websocket: WebSocket,
    reference_id: str = Query(..., description="서버에 등록된 기준 안무 ID"),
):
    await websocket.accept()
    reference = settings.reference_keypoints.get(reference_id)
    if not reference:
        await websocket.send_json({"error": "등록되지 않은 reference_id입니다"})
        await websocket.close(code=4004)
        return

    try:
        video_id = int(reference["video_id"])
        keypoint_path = str(reference["keypoint_path"])
        fps = float(reference.get("fps", 30.0))
    except (KeyError, TypeError, ValueError):
        await websocket.send_json({"error": "reference 설정이 올바르지 않습니다"})
        await websocket.close(code=4004)
        return

    logger.info("WS 연결: reference_id=%s", reference_id)
    redis = await get_redis()
    try:
        reference_kp = await load_reference_keypoints(redis, video_id, keypoint_path)
    except Exception:
        logger.exception("reference keypoint 로드 실패")
        await websocket.send_json({"error": "keypoint 로드 실패"})
        await websocket.close(code=4003)
        return

    await websocket.send_json({"status": "ready", "video_id": video_id})

    warn_sec = _WARN_SEC
    pause_sec = _PAUSE_SEC

    last_feedback: dict | None = None
    last_recv = asyncio.get_event_loop().time()
    warn_sent = False
    user_kp_window: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
    ref_kp_window: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
    ref_idx_window: deque[int] = deque(maxlen=WINDOW_SIZE)
    processed_frame_count = 0
    try:
        while True:
            try:
                message = await asyncio.wait_for(_receive_one(websocket), timeout=warn_sec)
            except json.JSONDecodeError:
                # 프레임 하나가 손상돼도 세션 전체를 끊지 않고 에러만 전달한다.
                await websocket.send_json({"error": "잘못된 JSON 형식입니다"})
                continue
            except asyncio.TimeoutError:
                elapsed = asyncio.get_event_loop().time() - last_recv
                if elapsed >= pause_sec:
                    # 10초 이상 수신 없음 → 일시정지 권고
                    payload: dict = {
                        "warning": "연결이 불안정합니다. 잠시 멈춰 주세요.",
                        "recommend_pause": True,
                    }
                    if last_feedback:
                        payload["score"] = last_feedback["score"]
                        payload["frame_idx"] = last_feedback["frame_idx"]
                    await websocket.send_json(payload)
                    last_recv = asyncio.get_event_loop().time()  # 스팸 방지용 리셋
                    warn_sent = False
                elif not warn_sent:
                    # 3초 이상 수신 없음 → 마지막 점수 유지 + 경고만 전달
                    payload = {"warning": "서버 수신 지연 중입니다. 잠시 후 재개됩니다."}
                    if last_feedback:
                        payload["score"] = last_feedback["score"]
                        payload["frame_idx"] = last_feedback["frame_idx"]
                    await websocket.send_json(payload)
                    warn_sent = True
                continue

            if message is None:  # 연결 종료
                break

            last_recv = asyncio.get_event_loop().time()
            warn_sent = False

            frame_idx: int = message.get("frame_idx", 0)
            timestamp_ms: int | None = message.get("timestamp_ms")
            raw_kp: list = message.get("keypoints")  # shape: (33, 3)

            if raw_kp is None:
                await websocket.send_json({"error": "keypoints 필드 누락"})
                continue

            # ragged 배열(관절마다 좌표 개수가 다른 경우)은 np.array 생성 자체에서
            # ValueError가 나고, 관절 개수가 다르면 compute_feedback의 np.dot에서 난다.
            # 둘 다 세션 전체를 끊기지 않도록 여기서 걸러낸다.
            try:
                incoming_kp = np.array(raw_kp, dtype=np.float32)  # (33, 3)
                shape_ok = incoming_kp.shape == (NUM_LANDMARKS, KEYPOINT_DIM)
            except ValueError:
                shape_ok = False
            if not shape_ok:
                await websocket.send_json({
                    "error": f"keypoints shape이 올바르지 않습니다 (기대: [{NUM_LANDMARKS}, {KEYPOINT_DIM}])"
                })
                continue
            feedback = compute_feedback(
                reference_kp,
                incoming_kp,
                frame_idx,
                timestamp_ms=timestamp_ms,
                fps=fps,
            )
            last_feedback = feedback
            processed_frame_count += 1
            ref_idx = feedback["frame_idx"] % reference_kp.shape[0]
            user_kp_window.append(incoming_kp)
            ref_kp_window.append(reference_kp[ref_idx])
            ref_idx_window.append(ref_idx)
            if (
                len(user_kp_window) == WINDOW_SIZE
                and processed_frame_count % WINDOW_SIZE == 0
                and not window_has_reference_seam(list(ref_idx_window), reference_kp.shape[0])
            ):
                try:
                    feedback["pentagon_scores"] = await asyncio.to_thread(
                        score_window, list(user_kp_window), list(ref_kp_window)
                    )
                except Exception:
                    logger.exception("pentagon 채점 실패: reference_id=%s", reference_id)
            await websocket.send_json(feedback)

    except WebSocketDisconnect:
        logger.info("WS 종료: reference_id=%s", reference_id)
    except Exception:
        logger.exception("WS 처리 중 오류: reference_id=%s", reference_id)


async def _receive_one(websocket: WebSocket) -> dict | None:
    """text JSON 메시지 1건 수신. 연결 종료 시 None 반환."""
    msg = await websocket.receive()
    if msg["type"] == "websocket.disconnect":
        return None
    text = msg.get("text")
    if text is None:
        # 바이너리 프레임은 지원하지 않음 - JSONDecodeError와 동일하게 처리해
        # 세션을 끊지 않고 에러만 전달한다.
        raise json.JSONDecodeError("binary frame not supported", "", 0)
    return json.loads(text)
