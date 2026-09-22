# Groovo Pentagon Scoring Handoff v2

## 입력 데이터

- user_seq: (30, 33, 3)
- idol_seq: (30, 33, 3)
- MediaPipe Pose 33개 관절의 x, y, z 좌표

## 오각형 점수

- Accuracy: 30%
- Detail: 15%
- Balance: 20%
- Timing: 15%
- Rhythm: 20%

## 최종 점수 공식

rule_score =
    Accuracy x 0.30
  + Detail x 0.15
  + Balance x 0.20
  + Timing x 0.15
  + Rhythm x 0.20

현재 final_score는 rule_score와 동일합니다.

DCA 점수는 dca_score 필드로 별도 전달할 수 있지만,
현재 final_score 계산에는 포함하지 않습니다.

## v2 변경 사항

- 실제 계산에 사용되지 않는 PentagonConfig 필드 제거
- 구형 Balance 3항목 설명 제거
- 현재 5-component Balance 공식으로 문서 통일
- 실제 점수 계산 방식과 기본 점수 결과는 변경 없음
- 로컬 전체 회귀 테스트 359개 통과
- 코랩 통합 테스트 111개 통과

## 실행 가능한 샘플 JSON

json/sample_request.json은 입력 검증과 점수 계산을
실제로 통과하는 30x33x3 더미 동작입니다.

사용자 동작에는 3프레임 지연이 적용되어 있습니다.

테스트 실행:

python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
