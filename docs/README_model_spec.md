# Groovo DCA-Net v2 Prototype

## 1. 프로젝트 목적

Groovo DCA-Net v2 Prototype은 사용자의 춤 동작과 기준 아이돌 안무 동작을 비교하여 점수와 시각화용 피드백을 생성하는 딥러닝 모델 프로토타입이다.

v2는 PDF 기준 DCA-Net 구조를 반영하는 33관절 모델이다. 입력은 MediaPipe Pose의 33개 관절을 사용하며, `user_seq`와 `idol_seq`를 함께 받아 두 동작 사이의 차이를 학습하는 것을 목표로 한다.

이번 프로젝트는 기존 17관절 GRU baseline과 섞지 않고 별도 프로젝트로 관리한다. 기존 v1 baseline은 비교 기준으로 남긴다.

## 2. v1 baseline과 v2 DCA-Net의 차이

기존 v1 baseline은 17관절 기반 GRU 모델을 중심으로 구성된 비교 기준 모델이다.

v2 DCA-Net은 다음 차이를 가진다.

- MediaPipe Pose 33관절 사용
- 입력 길이 30프레임 고정
- `user_seq`, `idol_seq` 각각 `(30, 33, 3)` 입력
- 공간 구조, 시간 흐름, 사용자-아이돌 간 대응 관계를 분리해서 처리
- Choreography Cross-Attention을 통해 기준 안무와 사용자 동작의 관계를 직접 비교
- 점수뿐 아니라 관절별 하이라이트 생성을 목표로 함

## 3. 입력 데이터 형식

모델 입력은 다음 두 개의 sequence이다.

- `user_seq`: 사용자 포즈 sequence
- `idol_seq`: 기준 아이돌 포즈 sequence

각 입력 shape는 다음과 같다.

```text
(30, 33, 3)
```

의미는 다음과 같다.

```text
30 = 최근 30프레임
33 = MediaPipe Pose 33개 관절
3  = x, y, z 또는 x, y, confidence
```

정규화는 다음 원칙을 기본으로 한다.

- 골반 중심 정렬
- 어깨너비 기준 scaling

골반 중심은 left hip과 right hip의 중간점을 사용한다.
scale 기준은 left shoulder와 right shoulder 사이의 거리를 사용한다.

## 4. 모델 목표 구조

DCA-Net v2의 목표 구조는 다음 흐름을 따른다.

```text
Input & Normalization
-> Spatial Encoder
-> Temporal Encoder
-> Choreography Cross-Attention
-> Feedback Decoder
-> Output Head
```

각 단계의 역할은 다음과 같다.

- Input & Normalization: 사용자와 아이돌 포즈를 같은 기준 좌표계로 정렬
- Spatial Encoder: 한 프레임 안에서 관절들의 자세 구조를 읽음
- Temporal Encoder: 30프레임 동안 움직임의 흐름을 읽음
- Choreography Cross-Attention: 사용자 동작과 기준 안무의 대응 관계를 비교
- Feedback Decoder: 어떤 관절 또는 좌표가 어긋났는지 피드백 정보 생성
- Output Head: 최종 점수와 하이라이트 정보를 생성

## 5. 출력 데이터 형식

최종 출력 목표는 다음과 같다.

```json
{
  "score_100": 87.3,
  "color": "green",
  "highlight_joints": [11, 13, 15],
  "highlight_coords": [[0.51, 0.33, 0.0], [0.48, 0.42, 0.0], [0.46, 0.55, 0.0]],
  "joint_errors": [0.01, 0.03, 0.02]
}
```

`score_100`은 모델 출력이다.

`color`는 모델이 직접 분류하지 않고 score 기준 후처리로 만든다.

```text
score_100 >= 80        -> green
60 <= score_100 < 80   -> orange
score_100 < 60         -> red
```

`highlight_joints`와 `highlight_coords`는 초기에는 관절별 오차 후처리로 생성한다. 이후 Feedback Decoder를 고도화하여 더 정교한 피드백을 생성하는 방향으로 확장한다.

## 6. 실시간 추론 방식

실시간 서비스에서는 서버가 최근 30프레임을 buffer에 저장한다. 새 프레임이 들어올 때마다 오래된 프레임을 제거하고 최신 프레임을 추가하는 sliding window 방식으로 모델을 호출한다.

예상 흐름은 다음과 같다.

```text
client camera
-> MediaPipe Pose
-> 33-joint pose frame
-> server frame buffer
-> recent 30-frame user_seq
-> matched 30-frame idol_seq
-> DCA-Net v2 inference
-> score and highlight response
```

## 7. 현재 구현 상태

현재 DCA-Net v2 prototype은 다음 구성까지 구현되었다.

- `model.py`: DCA-Net v2 모델 구조
- `dataset.py`: PyTorch Dataset / DataLoader 구성
- `train.py`: 학습 루프
- `evaluate.py`: train / valid / test 평가
- `inference.py`: checkpoint 기반 추론
- `realtime_buffer.py`: 30프레임 sliding window buffer
- `service.py`: 서버 연동용 wrapper

현재 모델은 실제 사람 채점 라벨이 아니라 rule-based pseudo label 기반 prototype이다.

## 8. 향후 작업

향후 작업은 다음과 같다.

1. 실제 사용자 영상 데이터 확보
2. 사람이 평가한 실제 score label 구축
3. L/R/P view를 활용한 multi-view 확장
4. WebSocket 기반 실시간 서버와 연동
5. 코사인 유사도 기반 기존 추론 결과와 DCA-Net v2 결과 비교
6. highlight_joints를 프론트엔드 overlay와 연결