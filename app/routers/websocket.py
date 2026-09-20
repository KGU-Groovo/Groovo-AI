import asyncio
import json
import logging
import time
from collections import deque

import numpy as np
import redis.asyncio as aioredis
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.config import settings
from app.redis_client import get_redis
from app.services.keypoint_service import (
    KEYPOINT_DIM,
    NUM_LANDMARKS,
    compute_feedback,
    load_reference_keypoints,
    upload_session_detail,
)
from app.services.pentagon_scoring_service import (
    WINDOW_SIZE,
    aggregate_pentagon_results,
    score_window,
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
        # 세션 키가 아예 없으면 _finalize_session의 ttl==-2 가드가 알아서
        # 아무 것도 안 하고, 키는 있는데 데이터가 깨진 경우엔 finished로
        # 갱신해서 active로 방치되지 않게 한다.
        await _finalize_session(redis, session_id, [], [])
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
        # 세션 자체는 Redis에 존재하는 상태(2단계 통과)이므로, 여기서 끝내지
        # 않으면 active 상태로 TTL 만료 때까지 방치된다. 프레임은 하나도
        # 처리 못 했으니 summary 없이 상태만 finished로 갱신한다.
        await _finalize_session(redis, session_id, [], [])
        return

    await websocket.send_json({"status": "ready", "video_id": session.video_id})

    warn_sec = _WARN_SEC
    pause_sec = _PAUSE_SEC

    last_feedback: dict | None = None
    last_recv = asyncio.get_event_loop().time()
    warn_sent = False
    frame_details: list[dict] = []
    # pentagon_scoring용 슬라이딩 윈도우 버퍼 (Notion "웹소켓 연결" 문서의 DCA-Net v2
    # 연동 설계와 동일한 구조: 최근 WINDOW_SIZE프레임만 유지하다가 30프레임마다 보조
    # 채점을 실행한다). 매 프레임 실시간 응답 경로(compute_feedback)와는 무관하다.
    user_kp_window: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
    ref_kp_window: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
    pentagon_window_results: list[dict] = []
    processed_frame_count = 0

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
            frame_details.append({
                "frame_idx": feedback["frame_idx"],
                "timestamp_ms": timestamp_ms,
                "score": feedback["score"],
                "worst_joints": feedback["worst_joints"],
            })
            await websocket.send_json(feedback)

            processed_frame_count += 1
            user_kp_window.append(incoming_kp)
            ref_kp_window.append(reference_kp[feedback["frame_idx"] % reference_kp.shape[0]])
            # 30프레임 단위 보조 추론 (Notion 문서: 매 프레임 필수 경로에 넣지 않고
            # 30프레임마다만 실행 + to_thread로 실시간 응답 루프를 막지 않는다).
            if len(user_kp_window) == WINDOW_SIZE and processed_frame_count % WINDOW_SIZE == 0:
                try:
                    window_result = await asyncio.to_thread(
                        score_window, list(user_kp_window), list(ref_kp_window)
                    )
                    pentagon_window_results.append(window_result)
                except Exception:
                    logger.exception(
                        "pentagon 오각형 점수 계산 실패 (frame %s): session_id=%s",
                        processed_frame_count,
                        session_id,
                    )

    except WebSocketDisconnect:
        logger.info("WS 종료: session_id=%s", session_id)
    except Exception:
        logger.exception("WS 처리 중 오류: session_id=%s", session_id)
    finally:
        await _finalize_session(redis, session_id, frame_details, pentagon_window_results)


async def _finalize_session(
    redis: aioredis.Redis,
    session_id: str,
    frame_details: list[dict],
    pentagon_window_results: list[dict],
) -> None:
    """세션 종료 처리 (Notion '웹소켓 연결' 문서 '분석 종료 처리' 참고).

    session:{id} 상태를 finished로 갱신하고, 처리한 프레임이 있으면
    session:{id}:summary에 평균 점수와 프레임별 상세 데이터의 S3 경로를
    저장한다. Spring Boot가 이 키를 읽어 reports 레코드를 생성한다.
    """
    key = f"session:{session_id}"
    ttl = await redis.ttl(key)

    if ttl == -2:
        # 세션 키가 이미 TTL 만료로 사라짐. 여기서 HSET을 하면 TTL 없이
        # 새로 생성돼(HSET은 키를 자동 생성함) 영구히 안 지워지는 좀비 키가
        # 되므로, 이미 사라진 세션은 갱신하지 않는다.
        logger.warning("세션 종료 처리 시점에 session:%s가 이미 만료됨", session_id)
    else:
        await redis.hset(
            key, mapping={"status": "finished", "finished_at": str(int(time.time()))}
        )

    if not frame_details:
        return

    detail_path: str | None = None
    try:
        detail_path = await upload_session_detail(session_id, frame_details)
    except Exception:
        # 상세 결과 업로드가 실패해도 평균 점수는 남겨야 하므로 종료 처리를
        # 막지 않는다. detail_path는 None으로 남는다.
        logger.exception("세션 상세 결과 S3 업로드 실패: session_id=%s", session_id)

    pentagon_scores = aggregate_pentagon_results(pentagon_window_results)

    scores = [frame["score"] for frame in frame_details]
    summary = {
        "average_score": round(sum(scores) / len(scores), 4),
        "frame_count": len(frame_details),
        "detail_path": detail_path,
        "pentagon_scores": pentagon_scores,
    }
    summary_ttl = ttl if ttl and ttl > 0 else settings.redis_session_ttl
    await redis.set(f"{key}:summary", json.dumps(summary), ex=summary_ttl)


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
