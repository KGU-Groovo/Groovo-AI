import logging

import numpy as np

from app.pentagon.pentagon_scores import PENTAGON_SCORE_NAMES, compute_pentagon_scores

logger = logging.getLogger(__name__)

# pentagon_scoring은 (30, 33, 3) 고정 윈도우를 기대한다 (PentagonConfig.expected_frames).
# Notion "웹소켓 연결"/"추론" 문서의 DCA-Net v2 연동 설계(슬라이딩 윈도우 버퍼, 30프레임
# 단위 보조 추론)를 pentagon_scoring에도 동일하게 적용한다.
WINDOW_SIZE = 30


def window_has_reference_seam(ref_indices: list[int], num_frames: int) -> bool:
    """윈도우 안에서 기준 영상이 순환(loop)되는 이음매를 지나가는지 검사한다.

    기준 영상이 WINDOW_SIZE보다 짧으면 슬라이딩 버퍼가 기준 프레임을
    `% num_frames`로 순환시키는데, pentagon_scoring은 입력이 연속된 한 클립이라고
    가정하므로 "마지막 프레임 → 다시 처음 프레임"으로 튀는 이 이음매를 실제
    동작으로 오인해 정렬(timing lag 탐색)이 깨지고 accuracy/balance가 크게
    잘못 나온다 (완벽히 일치하는 데이터로도 accuracy=0이 나오는 걸 실측 확인함).
    이런 윈도우는 계산 자체를 스킵하는 게 잘못된 점수를 내는 것보다 안전하다.

    순환하지 않는 정상 구간에서는 인덱스가 (fps 차이로 가끔 반복될 수는 있어도)
    큰 폭으로 뒤로 가지 않으므로, num_frames의 절반보다 큰 역방향 점프만
    이음매로 판단한다.
    """
    if num_frames <= 1:
        return False
    half = num_frames / 2
    return any(
        prev - curr > half for prev, curr in zip(ref_indices, ref_indices[1:])
    )


def score_window(user_window: list[np.ndarray], ref_window: list[np.ndarray]) -> dict:
    """최근 WINDOW_SIZE프레임(사용자/기준)으로 오각형 점수 1회 계산.

    호출 측(websocket.py)이 30프레임마다 asyncio.to_thread로 호출해 실시간 응답
    경로를 막지 않도록 한다 (Notion 문서 "성능 및 지연시간 관점": DCA류 보조 추론은
    매 프레임 필수 경로에 넣지 않는다).
    """
    user_seq = np.stack(user_window)
    idol_seq = np.stack(ref_window)
    result = compute_pentagon_scores(user_seq, idol_seq)
    return {
        "final_score": round(result.final_score, 4),
        "scores": {name: round(getattr(result.scores, name), 4) for name in PENTAGON_SCORE_NAMES},
    }


def aggregate_pentagon_results(window_results: list[dict]) -> dict | None:
    """세션 동안 30프레임마다 계산된 오각형 점수들을 평균 내어 리포트용으로 요약.

    한 번도 30프레임에 도달하지 못한 세션은 None을 반환한다.
    """
    if not window_results:
        return None

    component_avgs = {
        name: round(
            float(np.mean([r["scores"][name] for r in window_results])), 4
        )
        for name in PENTAGON_SCORE_NAMES
    }
    final_score = round(
        float(np.mean([r["final_score"] for r in window_results])), 4
    )

    return {
        "final_score": final_score,
        "scores": component_avgs,
        "window_count": len(window_results),
    }
