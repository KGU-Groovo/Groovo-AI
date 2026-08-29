import asyncio
import json
import logging

import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.redis_client import get_redis
from app.services.keypoint_service import (
    KEYPOINT_DIM,
    NUM_LANDMARKS,
    compute_feedback,
    load_reference_keypoints,
)
from app.services.session_service import get_session
from app.services.token_service import verify_ws_token

router = APIRouter()
logger = logging.getLogger(__name__)

_WARN_SEC = 3.0    # 경고 임계값: 3초 동안 수신 없으면 마지막 점수 유지 + 경고
_PAUSE_SEC = 10.0  # 일시정지 권고 임계값: 10초


@router.websocket("/ws/analyze")
async def analyze_websocket(
    websocket: WebSocket,
    token: str = Query(..., description="Spring Boot가 발급한 ws_token"),
):
    # accept()는 함수 전체에서 한 번만 호출한다. accept() 전에 close()를 호출하면
    # WS close frame이 아니라 HTTP 레벨 거부(403)로 전달되어 클라이언트가 커스텀
    # 종료 코드(4001/4002/4003)를 받지 못한다. 이후 모든 거부 경로는 accept() 이후에
    # 동일한 패턴(에러 메시지 전송 → close(code=...))으로 처리한다.
    await websocket.accept()

    # 1. 토큰 검증
    try:
        token_payload = verify_ws_token(token)
    except ValueError as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close(code=4001, reason=str(e))
        return

    session_id = token_payload.session_id
    logger.info("WS 연결: session_id=%s user_id=%s", session_id, token_payload.user_id)

    redis = await get_redis()

    # 2. Redis에서 세션 메타데이터 조회
    try:
        session = await get_session(redis, session_id)
    except LookupError as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close(code=4002)
        return

    # 3. 기준 keypoint 로드 (Redis 캐시 → S3 순서)
    try:
        reference_kp = await load_reference_keypoints(
            redis, session.video_id, session.keypoint_path
        )
    except Exception:
        logger.exception("reference keypoint 로드 실패")
        await websocket.send_json({"error": "keypoint 로드 실패"})
        await websocket.close(code=4003)
        return

    await websocket.send_json({"status": "ready", "video_id": session.video_id})

    warn_sec = _WARN_SEC
    pause_sec = _PAUSE_SEC

    last_feedback: dict | None = None
    last_recv = asyncio.get_event_loop().time()
    warn_sent = False

    # 4. 실시간 keypoint 수신 → 비교 → 피드백 반환
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
                fps=session.fps,
            )
            last_feedback = feedback
            await websocket.send_json(feedback)

    except WebSocketDisconnect:
        logger.info("WS 종료: session_id=%s", session_id)
    except Exception:
        logger.exception("WS 처리 중 오류: session_id=%s", session_id)


async def _receive_one(websocket: WebSocket) -> dict | None:
    """text JSON 메시지 1건 수신. 연결 종료 시 None 반환."""
    msg = await websocket.receive()
    if msg["type"] == "websocket.disconnect":
        return None
    return json.loads(msg["text"])
