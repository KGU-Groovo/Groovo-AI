# DCA-Net v2 Model Design

이 문서는 DCA-Net v2의 초기 설계 기준을 설명한다.
현재 구현된 model.py는 이 설계 방향을 기반으로 작성되었으며,
세부 구현은 src/model.py를 기준으로 확인한다.

## 1. 전체 구조

DCA-Net v2의 목표 흐름은 다음과 같다.

```text
Input & Normalization
-> Spatial Encoder
-> Temporal Encoder
-> Choreography Cross-Attention
-> Feedback Decoder
-> Output Head
```

입력은 사용자 동작과 기준 아이돌 동작이다.

```text
user_seq: (30, 33, 3)
idol_seq: (30, 33, 3)
```

여기서 30은 프레임 수, 33은 MediaPipe Pose 관절 수, 3은 좌표 차원이다.
좌표는 x, y, z 또는 x, y, confidence 형식을 사용할 수 있다.

## 2. Input & Normalization

입력 포즈는 카메라 위치, 사람의 키, 화면 안 위치에 따라 값이 달라질 수 있다.
따라서 모델에 넣기 전에 기준을 맞춰야 한다.

v2의 기본 정규화 원칙은 다음과 같다.

1. 골반 중심 정렬
2. 어깨너비 기준 scaling

골반 중심 정렬은 left hip과 right hip의 중간점을 기준으로 한다.
모든 관절 좌표에서 이 중심점을 빼면, 사람의 위치가 화면 어디에 있든 골반을 중심으로 한 상대 좌표가 된다.

어깨너비 기준 scaling은 left shoulder와 right shoulder 사이의 거리를 사용한다.
좌표를 이 거리로 나누면, 키나 카메라 거리 차이에 의한 크기 차이를 줄일 수 있다.

MediaPipe Pose 기준 주요 index는 다음과 같다.

```text
left shoulder  = 11
right shoulder = 12
left hip       = 23
right hip      = 24
```

## 3. Spatial Encoder

Spatial Encoder는 한 프레임 안에서 관절들이 어떤 자세를 이루는지 읽는 부분이다.

예를 들어 팔이 위로 올라갔는지, 무릎이 굽혀졌는지, 상체가 기울어졌는지 같은 정보는 한 프레임 안의 관절 배치를 보면 알 수 있다.

입력 예시는 다음과 같다.

```text
한 프레임: (33, 3)
```

Spatial Encoder는 33개 관절의 관계를 학습하여 각 프레임의 자세 특징을 만든다.

초기 구현에서는 이해하기 쉬운 Transformer 기반 구조로 시작한다.
이후 필요하면 관절 연결 관계를 반영하는 구조로 확장한다.

## 4. Temporal Encoder

Temporal Encoder는 30프레임 동안 동작이 어떻게 변하는지 읽는 부분이다.

춤은 한 장의 자세만으로 판단하기 어렵다.
팔을 올리는 과정, 스텝의 타이밍, 몸의 이동 방향처럼 시간 흐름이 중요하다.

Temporal Encoder는 Spatial Encoder가 만든 프레임별 특징을 받아서 30프레임 전체의 움직임 특징을 만든다.

```text
프레임별 자세 특징 30개
-> 시간 흐름 인코딩
-> 동작 sequence 특징
```

## 5. Choreography Cross-Attention

Choreography Cross-Attention은 사용자 동작과 기준 아이돌 동작을 비교하는 핵심 부분이다.

쉬운 말로 하면, 모델이 다음 질문을 하도록 만드는 구조다.

```text
사용자의 이 동작은 기준 안무의 어느 부분과 비교해야 하는가?
어느 프레임, 어느 관절에서 차이가 큰가?
```

단순히 user_seq와 idol_seq를 빼는 방식은 타이밍 차이나 동작 흐름을 충분히 반영하기 어렵다.
Cross-Attention은 사용자 특징이 기준 안무 특징을 참고하면서 차이를 찾게 한다.

초기 구현에서는 다음 형태를 목표로 한다.

```text
user temporal features
idol temporal features
-> cross-attention
-> comparison features
```

## 6. Feedback Decoder

Feedback Decoder는 비교 결과를 프론트에서 사용할 수 있는 시각화 정보로 바꾸는 부분이다.

최종적으로는 다음 정보를 만들고자 한다.

- 어떤 관절이 많이 틀렸는지
- 어느 좌표 방향에서 차이가 큰지
- 사용자에게 강조해서 보여줄 관절은 어디인지

초기 버전에서는 Feedback Decoder를 복잡하게 만들지 않고, 관절별 오차를 후처리해서 `highlight_joints`, `highlight_coords`, `joint_errors`를 만든다.

이후 고도화 단계에서는 Feedback Decoder가 직접 피드백 특징을 학습하도록 확장한다.

## 7. Output Head

Output Head는 모델의 최종 출력을 만드는 부분이다.

v2의 출력 목표는 다음과 같다.

```json
{
  "score_100": 87.3,
  "color": "green",
  "highlight_joints": [11, 13, 15],
  "highlight_coords": [[0.51, 0.33, 0.0], [0.48, 0.42, 0.0], [0.46, 0.55, 0.0]],
  "joint_errors": [0.01, 0.03, 0.02]
}
```

`score_100`은 모델이 직접 출력하는 0~100 점수다.

`color`는 모델이 직접 분류하지 않는다.
점수 기준 후처리로 만든다.

```text
score_100 >= 80        -> green
60 <= score_100 < 80   -> orange
score_100 < 60         -> red
```

`highlight_joints`, `highlight_coords`, `joint_errors`는 초기에는 관절별 오차 기반으로 만들고, 이후 Feedback Decoder를 통해 더 정교하게 만든다.

## 8. 서버 연동 예상 흐름

실시간 서비스에서는 클라이언트 또는 서버에서 MediaPipe Pose를 사용해 프레임마다 33개 관절 좌표를 얻는다.

서버는 최근 30프레임을 buffer에 저장한다.
새 프레임이 들어오면 buffer를 갱신하고, 30프레임이 모이면 모델 추론을 수행한다.

예상 흐름은 다음과 같다.

1. 사용자 영상 입력
2. MediaPipe Pose로 33관절 추출
3. 서버가 최근 30프레임 buffer 유지
4. 기준 안무 `idol_seq`와 현재 `user_seq` 준비
5. 정규화 수행
6. DCA-Net v2 모델 추론
7. `score_100` 출력
8. `score_100` 기준 color 후처리
9. 관절별 오차 기반 highlight 생성
10. 클라이언트에 결과 반환

이 방식은 sliding window 방식이다.
즉, 매번 완전히 새로운 30프레임을 기다리는 것이 아니라, 최근 프레임들을 계속 밀어가며 추론한다.

## 9. v2에서 우선 구현할 것

초기 v2 prototype에서 우선 구현할 것은 다음과 같다.

1. 설정 파일과 문서 기준 고정
2. 33관절 입력 shape 검증
3. 골반 중심 정렬
4. 어깨너비 scaling
5. Spatial Encoder 기본 구조
6. Temporal Encoder 기본 구조
7. Cross-Attention 기본 구조
8. `score_100` 출력
9. score 기준 color 후처리
10. 관절별 오차 기반 highlight 생성

## 10. 나중에 고도화할 것

초기 prototype 이후 고도화할 항목은 다음과 같다.

1. Feedback Decoder를 별도 학습 모듈로 강화
2. 관절 연결 관계를 반영한 Spatial Encoder 개선
3. 박자와 음악 beat 정보를 함께 사용하는 구조 검토
4. 동작 구간별 점수 계산
5. 사람마다 다른 체형과 카메라 각도에 더 강한 정규화
6. 서버 실시간 추론 최적화
7. 모바일 또는 웹 클라이언트용 응답 포맷 확정