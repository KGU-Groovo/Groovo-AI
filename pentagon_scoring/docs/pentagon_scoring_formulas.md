# 규칙 기반 오각형 점수 계산식 초기 명세 (11-1)

## 1. 범위와 공통 원칙

이 문서는 오각형 점수 계산식의 초기안이며 11-2 공통 기반, 11-3 timing, 11-4 accuracy 및 11-5 detail 구현 결정을 반영했다. `score_detail.py`와 `test_score_detail.py`에서 말단 위치·관절 각도 detail 계산을 구현했지만 balance, rhythm 및 종합 점수 실행 코드는 아직 없다. 여기의 상수와 가중치는 사람이 직접 채점한 정답 점수를 학습해 얻은 값이 아니라 **MVP 초기 calibration 값**이며, 설명용 숫자 예시는 실제 calibration 결과가 아니다.

오각형 점수는 timing, accuracy, detail, balance, rhythm으로 구성한다. 모든 점수는 원칙적으로 `[0, 100]`으로 clipping한다.

```text
clip(x, 0, 100) = min(100, max(0, x))
```

종합 정책은 다음과 같다.

```text
rule_score =
    0.30 * accuracy
  + 0.15 * detail
  + 0.20 * balance
  + 0.15 * timing
  + 0.20 * rhythm

final_score = rule_score
score_mode = "rule_based"
```

DCA-Net 점수는 `dca_score`라는 보조 점수로 분리하고 현재는 혼합하지 않는다.

## 2. 사용 가능한 관절 index

현재 코드는 33개 관절 입력을 사용하지만 landmark 이름/index mapping을 정의하지 않았다. 아래는 향후 입력이 **MediaPipe Pose 33 landmarks임이 확인될 경우** 사용할 표준 index 후보이다.

| Index | Landmark | Index | Landmark |
|---:|---|---:|---|
| 0 | nose | 17 | left_pinky |
| 1 | left_eye_inner | 18 | right_pinky |
| 2 | left_eye | 19 | left_index |
| 3 | left_eye_outer | 20 | right_index |
| 4 | right_eye_inner | 21 | left_thumb |
| 5 | right_eye | 22 | right_thumb |
| 6 | right_eye_outer | 23 | left_hip |
| 7 | left_ear | 24 | right_hip |
| 8 | right_ear | 25 | left_knee |
| 9 | mouth_left | 26 | right_knee |
| 10 | mouth_right | 27 | left_ankle |
| 11 | left_shoulder | 28 | right_ankle |
| 12 | right_shoulder | 29 | left_heel |
| 13 | left_elbow | 30 | right_heel |
| 14 | right_elbow | 31 | left_foot_index |
| 15 | left_wrist | 32 | right_foot_index |
| 16 | right_wrist |  |  |

새 오각형 모듈은 세 번째 채널이 z인 `[x, y, z]`를 명시적 입력 계약으로 사용한다. 호출 측은 실제 입력이 이 계약을 만족함을 보장해야 하며 shape로 의미를 자동 추론하지 않는다. 기존 DCA prototype의 세 번째 채널은 실제 pose 추출 데이터가 없어 실제 MediaPipe z라고 확정할 수 없고, 새 계약은 기존 모델이 실제 z로 학습됐다는 증거가 아니다.

MediaPipe Pose 33에는 완전한 손가락 관절이 없다. 따라서 detail은 “손가락 디테일”이 아니라 손목과 `pinky`, `index`, `thumb` landmark 수준의 평가라고 표현해야 한다.

## 3. 전처리 조건

계산 전에 다음을 만족해야 한다.

1. `user_seq`와 `idol_seq`의 shape가 명시된 계약과 일치
2. numeric dtype이며 계산용 float로 안전하게 변환 가능
3. NaN과 Inf가 없음
4. 동일한 landmark 순서
5. 동일한 좌표계, 축 방향, 좌우 반전 정책
6. 동일한 좌표 정규화 방식
7. 유효 FPS 제공
8. 최소 공통 프레임 수 확보

11-2 공통 모듈은 다음 정규화를 구현한다.

```text
hip_center(t) =
  0.5 * (
      pose(t, left_hip)
    + pose(t, right_hip)
  )

centered_pose(t,j) =
  pose(t,j) - hip_center(t)

delta_shoulder_x(t) =
  right_shoulder_x(t) - left_shoulder_x(t)

delta_shoulder_y(t) =
  right_shoulder_y(t) - left_shoulder_y(t)

shoulder_width_xy(t) =
  sqrt(
      delta_shoulder_x(t)^2
    + delta_shoulder_y(t)^2
  )

sequence_scale =
  median(valid shoulder_width_xy)

normalized_pose(t,j) =
  centered_pose(t,j) / sequence_scale
```

- 골반 중심은 MediaPipe Pose 23/24로 계산하고 매 프레임 모든 관절에서 뺀다.
- 어깨너비는 landmarks 11/12의 x/y만 사용한다. 카메라 기반 z가 불안정할 수 있어 scale 계산에서는 z를 제외한다.
- 기본 30프레임이면 유효한 30개 어깨너비의 중앙값을 하나의 sequence scale로 사용한다.
- 프레임별 어깨너비로 나누지 않는다. 검출 노이즈로 인한 scale jitter를 줄이기 위해서다.
- 사용자와 기준 sequence는 체형과 촬영 크기 차이를 줄이기 위해 각각 독립적으로 scale한다.
- z도 프레임별 골반 z를 빼고 같은 `sequence_scale`로 나눈다. 점수 거리에서는 별도의 `z_weight`로 기여도를 낮출 수 있다.
- 좌우 반전, 회전, 원근 및 카메라 보정은 아직 하지 않으며 정면 촬영을 가정한다.
- 유효한 어깨너비가 없거나 중앙값이 최소값보다 작으면 입력 오류로 처리한다.

visibility/confidence와 누락 landmark 정책은 아직 확인 필요다.

timing을 먼저 구해 sequence를 정렬한 뒤 나머지 네 점수를 계산한다. 사용자가 몇 프레임 늦거나 빠른 한 가지 이유로 accuracy, detail, balance, rhythm까지 중복 감점되는 것을 줄이기 위해서다.

## 4. 공통 기호와 초기 상수

| 기호/상수 | 의미 | 초기 상태 |
|---|---|---|
| `T` | frame 수 | 예시는 30, 일반식은 가변 |
| `J` | joint 수 | MediaPipe 확인 시 33 |
| `fps` | 초당 frame | 입력값; 예시는 30 |
| `epsilon` | 0 나눗셈 방지 | MVP 값 확정 필요 |
| `z_weight` | 깊이축 제곱항 가중치 | JSON 예시는 0.3, calibration 필요 |
| `joint_weight_j` | 관절별 중요도 | 목록 확정 필요 |
| `max_lag_frames` | 탐색할 최대 lag | MVP 값 확정 필요 |
| `timing_penalty_per_frame` | frame당 timing 감점 | 예시는 5 |
| `accuracy_scale` | 거리 오차 감점 scale | 예시는 300 |
| `DETAIL_POSITION_SCALE` | 말단 위치 오차 scale | 11-5 MVP 초기값 350.0 |
| `DETAIL_ANGLE_SCALE` | degree 각도 오차 scale | 11-5 MVP 초기값 1.5 |
| `rhythm_mae_scale` | motion MAE 감점 scale | MVP 후보 25.0 |

모든 상수는 MVP 초기값이며 실제 사용자 영상과 사람이 평가한 점수로 calibration해야 한다.

## 5. Timing

### 목적

기준 안무 대비 사용자의 전체 동작이 몇 frame 선행 또는 지연되었는지 평가한다.

### 계산식

관절 `j`의 연속 frame 변화량을 이용해 frame별 motion signal을 만든다.

```text
delta_x(t,j) = x(t,j) - x(t-1,j)
delta_y(t,j) = y(t,j) - y(t-1,j)
delta_z(t,j) = z(t,j) - z(t-1,j)

motion(t,j) =
  sqrt(
      delta_x(t,j)^2
    + delta_y(t,j)^2
    + z_weight * delta_z(t,j)^2
  )

m_t = weighted_mean_over_joints(motion(t,j))
```

각 motion signal을 표준화한다.

```text
normalized_signal =
  (signal - mean(signal)) / (std(signal) + epsilon)
```

`[-max_lag_frames, +max_lag_frames]`에서 사용자와 기준 신호의 겹치는 구간별 Pearson correlation을 계산하고 최대인 lag `L`을 선택한다. 전체 signal에 `np.correlate`를 적용하는 방식이 아니라, 각 lag의 실제 overlap을 자르고 그 구간의 평균을 다시 제거한다. overlap 표준편차가 `epsilon` 이하이거나 값이 2개 미만이면 해당 correlation은 무효다. 유효 motion overlap은 최소 `minimum_aligned_frames - 1`개다.

```text
lag > 0:
  user_overlap = user_motion[lag:]
  idol_overlap = idol_motion[:-lag]

lag < 0:
  user_overlap = user_motion[:lag]
  idol_overlap = idol_motion[-lag:]

lag == 0:
  user_overlap = user_motion
  idol_overlap = idol_motion

pearson(a, b) =
  sum((a - mean(a)) * (b - mean(b)))
  / sqrt(sum((a - mean(a))^2) * sum((b - mean(b))^2))

L = argmax_lag pearson(user_overlap, idol_overlap)

timing_score =
  clip(
    100 - abs(L) * timing_penalty_per_frame,
    0,
    100
  )

lag_ms = L / fps * 1000
abs_lag_ms = abs(L) / fps * 1000
```

lag 후보는 결정적으로 `0, -1, 1, -2, 2, ...` 순서로 검색한다. correlation 차이가 `epsilon` 이하인 tie에서는 먼저 나온 후보를 유지하므로 lag 0, 더 작은 절댓값, 같은 절댓값이면 음수 순으로 우선한다.

부호 및 pose 정렬 계약은 다음과 같다.

- `L == 0`: 사용자와 기준의 전역 timing 정렬, `lag_direction="aligned"`, 전체 pose 사용
- `L > 0`: 사용자가 늦음, `lag_direction="user_delayed"`, `user_seq[L:]`와 `idol_seq[:-L]`
- `L < 0`: 사용자가 빠름, `lag_direction="user_ahead"`, `user_seq[:L]`와 `idol_seq[-L:]`

정렬 frame 수는 `T - abs(L)`이다. timing을 accuracy/detail/balance/rhythm보다 먼저 계산하면 하나의 전역 지연 때문에 나머지 항목까지 중복 감점되는 것을 줄일 수 있다. 정렬된 공통 frame 구간은 후속 단계에서 네 점수 모두에 사용할 예정이다.

### Constant signal fallback

- 둘 다 `std <= epsilon`: `L=0`, score 100, direction `unobservable`, correlation `None`, `reliable=false`, reason `both_constant`
- 한쪽만 `std <= epsilon`: `L=0`, score 0, direction `unobservable`, correlation `None`, `reliable=false`, reason `one_constant`
- 둘 다 non-constant: 정상 lag 검색. `reliable`은 최고 correlation이 0보다 큰지를 나타낸다.

둘 다 정지한 경우의 100점은 timing 일치가 증명됐다는 뜻이 아니라 관찰 가능한 위반이 없어 감점하지 않는 fallback이다. 정상 검색에서도 correlation이 낮지만 lag 절댓값이 작으면 timing 점수가 높을 수 있다. timing 점수는 동작 형태 유사도가 아니라 전역 frame lag만 감점하기 때문이다.

timing은 음악 beat 분석이 아니라 기준 pose motion signal과 사용자 pose motion signal의 전역 프레임 지연을 평가한다.

### Diagnostics

- `score`
- `lag_frames`
- `lag_ms`
- `abs_lag_ms`
- `lag_direction`
- `max_lag_frames`
- `penalty_per_frame`
- `correlation_at_best_lag`
- `original_frame_count`
- `aligned_frame_count`
- `motion_overlap_count`
- `user_motion_std`, `idol_motion_std`
- `user_constant_signal`, `idol_constant_signal`
- `reliable`
- `fallback_reason`

`max_lag_frames`와 `timing_penalty_per_frame`는 실제 사용자 영상과 사람 평가로 재조정해야 하는 MVP calibration 값이다. 기본 30프레임 window는 긴 지연, 긴 안무, window 경계의 움직임을 관찰하기 어렵다.

### 숫자 예시

```text
lag_frames = 3
penalty_per_frame = 5
timing_score = 100 - 3 * 5 = 85

fps = 30
lag_ms = 3 / 30 * 1000 = 100 ms
abs_lag_ms = abs(3) / 30 * 1000 = 100 ms
```

설명용 예시이며 실제 calibration 결과가 아니다.

### 점수가 낮아지는 대표 사례

- 동작 전체가 기준보다 여러 frame 늦음
- 동작 전체가 기준보다 여러 frame 빠름
- motion peak 시점이 기준과 지속적으로 어긋남

## 6. Accuracy

### 목적

timing 보정 후 전신 관절 위치가 기준 안무와 얼마나 가까운지 평가한다. 입력은 timing이 반환한 정규화·정렬 완료 `(T_aligned, 33, 3)` pose이며 `T_aligned`는 30이 아닐 수 있다. 원본 config는 유지하고 `expected_frames=None`인 복사본으로 검증하며, 재정규화나 timing 재계산은 하지 않는다.

### 계산식

정렬된 공통 frame 구간에서 다음 거리를 계산한다.

```text
dx(t,j) = user_x(t,j) - idol_x(t,j)
dy(t,j) = user_y(t,j) - idol_y(t,j)
dz(t,j) = user_z(t,j) - idol_z(t,j)

d(t,j) =
  sqrt(
      dx(t,j)^2
    + dy(t,j)^2
    + z_weight * dz(t,j)^2
  )

use_z = false이면:
d(t,j) = sqrt(dx(t,j)^2 + dy(t,j)^2)

per_joint_error(j) = mean_over_frames(d(t,j))

mean_joint_error = mean_over_frames_and_joints(d(t,j))

joint_weight_sum = sum_over_joints(joint_weight(j))

weighted_mean_joint_error =
  sum_over_joints(joint_weight(j) * per_joint_error(j))
  / joint_weight_sum

per_frame_weighted_error(t) =
  sum_over_joints(joint_weight(j) * d(t,j))
  / joint_weight_sum

weighted_contribution(j) =
  joint_weight(j) * per_joint_error(j)
  / joint_weight_sum

accuracy_score =
  clip(
    100 - weighted_mean_joint_error * accuracy_scale,
    0,
    100
  )
```

`mean(per_frame_weighted_error)`와 `sum(weighted_contribution)`은 모두 `weighted_mean_joint_error`와 같다. `worst_joint_indices`는 weighted contribution 내림차순으로 고르고 동률이면 낮은 index를 우선한다. 따라서 관절별 오차가 가장 큰 관절과 최종 감점 기여가 가장 큰 관절은 가중치 때문에 다를 수 있다.

33개 MediaPipe 관절을 모두 사용한다. 얼굴은 춤의 큰 자세와 주요 사지 형태에 비해 낮은 중요도 가중치를, 몸통과 주요 팔·다리 관절은 높은 가중치를 사용한다. `use_z=true`일 때만 z를 `z_weight`로 완화해 반영한다. 이 중요도와 깊이 정책은 MVP 가정이다.

timing 정렬 뒤 계산하는 이유는 동일한 전역 frame 지연을 timing과 accuracy 양쪽에서 중복 감점하지 않기 위해서다. accuracy는 실제 음악 beat 분석이나 DCA 예측과 무관한 규칙 기반 점수다.

### Diagnostics

- `score`
- `mean_joint_error`
- `weighted_mean_joint_error`
- `accuracy_scale`
- `use_z`
- `z_weight`
- `frame_count`, `joint_count`
- `joint_weight_sum`, `top_k`
- `per_joint_errors`
- `per_joint_weights`
- `per_joint_weighted_contributions`
- `per_frame_weighted_errors`
- `worst_joint_indices`
- `worst_joints`: index, name, mean_error, weight, weighted_contribution

### 숫자 예시

```text
weighted_mean_joint_error = 0.08
accuracy_scale = 300
accuracy_score = 100 - 0.08 × 300 = 76점

left_wrist mean error = 0.12
left_wrist weight = 1.10
joint_weight_sum = 30.0이라고 가정

left_wrist contribution =
  0.12 × 1.10 / 30.0
  = 0.0044
```

두 번째 예시의 실제 weight 합은 코드 설정값으로 계산한다. `accuracy_scale=300`은 이론적으로 확정된 값이 아니라 MVP 점수 범위 조정을 위한 초기 calibration 값이며 실제 사용자 영상과 사람 평가로 조정해야 한다. 관절 가중치 역시 같은 calibration 대상이다.

### 점수가 낮아지는 대표 사례

- timing은 맞지만 팔·다리의 전신 위치가 기준과 다름
- 여러 주요 관절이 지속적으로 기준에서 멀어짐
- 몸 크기/중심 정규화가 불일치함. 이는 동작 품질이 아닌 전처리 오류이므로 입력 검증에서 분리해야 함

## 7. Detail

### 목적

11-5에서 프로젝트 루트에 `score_detail.py`와 `test_score_detail.py`를
생성했다. public API는 `compute_detail_score(...)`이며 결과는
`DetailScoreResult`와 `DetailDiagnostics`로 구성되고 `to_dict()` 결과는
JSON 직렬화할 수 있다.

timing 정렬이 끝난 normalized pose에서 전신 평균이 놓치기 쉬운 말단 위치와 팔꿈치·무릎 굽힘 각도를 평가한다. 입력은 `(T_aligned, 33, 3)`이며 정렬 후 길이는 30이 아닐 수 있어 `expected_frames=None`인 config 복사본으로 검증한다. 재정규화와 timing 재계산은 하지 않는다.

### 관절 후보

- 위치: wrists 15/16, pinkies 17/18, indices 19/20, thumbs 21/22, ankles 27/28, heels 29/30, foot indices 31/32
- 위치 가중치: `(1.2, 1.2, 0.7, 0.7, 0.8, 0.8, 0.7, 0.7, 1.2, 1.2, 0.9, 0.9, 1.0, 1.0)`
- 각도 triplet:
  - 왼 팔꿈치 `(11, 13, 15)`
  - 오른 팔꿈치 `(12, 14, 16)`
  - 왼 무릎 `(23, 25, 27)`
  - 오른 무릎 `(24, 26, 28)`

손 영역은 완전한 손가락 skeleton이 아니라 MediaPipe Pose의 제한된 hand landmark 수준이다.

### 계산식

endpoint 위치 오차는 공통 거리 함수를 재사용한다. `use_z=true`이면 `sqrt(dx² + dy² + z_weight·dz²)`, false이면 `sqrt(dx² + dy²)`이다.

```text
per_position_joint_error(i) = mean_over_frames(distance(t,i))

weighted_position_error =
  sum(position_weight(i) * per_position_joint_error(i))
  / position_weight_sum

per_frame_position_error(t) =
  sum(position_weight(i) * distance(t,i))
  / position_weight_sum

position_contribution(i) =
  position_weight(i) * per_position_joint_error(i)
  / position_weight_sum

position_score =
  clip(
    100 - weighted_position_error * 350,
    0,
    100
  )
```

세 점 `A-B-C`에서 `B`의 각도는 벡터 `u=A-B`, `v=C-B`로 계산한다.

```text
angle_deg(A,B,C) =
  degrees(arccos(
    clip(
      dot(u,v) / ((norm(u) * norm(v)) + epsilon),
      -1,
      1
    )
  ))

pair_valid(t,k) = user_valid(t,k) AND idol_valid(t,k)
angle_error_deg(t,k) = abs(user_angle_deg(t,k) - idol_angle_deg(t,k))
```

각도는 0~180도 내각이므로 원형 wrap 없이 절댓값 차이를 쓴다. 양쪽 pose가 모두 valid인 frame만 평균한다. 특정 angle에 valid frame이 없으면 effective weight를 0으로 만들어 제외한다.

```text
weighted_angle_error_deg =
  sum(effective_angle_weight(k) * per_angle_error_deg(k))
  / effective_angle_weight_sum

angle_contribution(k) =
  effective_angle_weight(k) * per_angle_error_deg(k)
  / effective_angle_weight_sum

angle_score =
  clip(
    100 - weighted_angle_error_deg * 1.5,
    0,
    100
  )

detail_score =
  clip(0.60 * position_score + 0.40 * angle_score, 0, 100)
```

네 각도가 모두 invalid이면 angle component는 관찰 불가능하다. 이때 `angle_score=None`, `weighted_angle_error_deg=None`, `reliable=false`, `fallback_reason="all_angles_invalid"`로 반환하고 `detail_score=position_score`로 가용 component를 100% 재정규화한다. 일부만 invalid이면 나머지 valid angle만 사용한다.

worst position joint는 position contribution 내림차순이고 동률은 낮은 MediaPipe index 우선이다. worst angle은 angle contribution 내림차순이고 동률은 spec 정의 순서 우선이며 `min(top_k, 4)`개를 반환한다.

### Diagnostics

- `score`
- `position_score`
- `angle_score`
- `weighted_position_error`, `weighted_angle_error_deg`
- position/angle scale 및 component weight
- frame/position-joint/angle count, weight sum
- `use_z`, `z_weight`, `top_k`
- 위치별 error/weight/contribution, frame별 position error
- position index/name, worst position joints
- angle name/error/weight/valid-frame-count/contribution, worst angles
- `reliable`, `fallback_reason`

### 숫자 예시

```text
weighted_position_error = 0.06
DETAIL_POSITION_SCALE = 350
position_score = 100 - 0.06 × 350 = 79

weighted_angle_error_deg = 12
DETAIL_ANGLE_SCALE = 1.5
angle_score = 100 - 12 × 1.5 = 82

detail_score = 79 × 0.60 + 82 × 0.40
             = 80.2점
```

설명용 예시이며 실제 사람 평가 기반 calibration 결과가 아니다. position scale 350, angle scale 1.5, 위치·각도 가중치 및 60:40 비율은 모두 MVP 초기값으로 실제 사용자 영상과 사람 평가로 조정해야 한다. detail은 DCA 예측값이 아닌 규칙 기반 점수이며 실제 음악 beat와 무관하다. timing 정렬 후 계산하여 전역 지연을 중복 감점하지 않는다.

### 점수가 낮아지는 대표 사례

- 손목이나 발끝 방향/위치가 기준과 크게 다름
- 팔 전체 위치는 비슷하지만 팔꿈치 굽힘이 다름
- 무릎 굽힘의 크기나 시점이 정렬 후에도 다름

## 8. Balance

### 목적

사용자의 절대적인 “바른 자세”가 아니라 같은 frame의 기준 안무와 비교하여 상체·골반 기울기와 지지 중심 차이를 평가한다.

### 점수가 낮아지는 대표 사례

- 기준보다 어깨선이 크게 기울어짐
- 기준과 다른 방향/크기로 골반이 기울어짐
- 몸 중심이 기준보다 발 지지 중심 밖으로 크게 벗어남
- 카메라가 기울어진 경우도 낮아질 수 있으므로 촬영 조건 검증 필요

### 계산식

11-6에서 `score_balance.py`와 `test_score_balance.py`를 구현했다. 입력은
timing 정렬 후 normalized `(T_aligned, 33, 3)` pose이며, 원본 config를
변경하지 않고 `expected_frames=None`인 복사본으로 가변 frame 수를 검증한다.
`minimum_aligned_frames`는 유지한다. 재정규화와 timing 재계산은 하지 않는다.
정면 2D 카메라의 x, y만 사용하고 z는 사용하지 않는다.

```text
shoulder_center = (left_shoulder + right_shoulder) / 2
hip_center = (left_hip + right_hip) / 2
ankle_center = (left_ankle + right_ankle) / 2

shoulder_vector = right_shoulder_xy - left_shoulder_xy
hip_vector = right_hip_xy - left_hip_xy
torso_vector = shoulder_center_xy - hip_center_xy
angle(v) = degrees(atan2(v_y, v_x))

raw = abs(a - b) % 180
axial_difference(a,b) = min(raw, 180 - raw)
circular_difference(a,b) = abs(((a - b + 180) % 360) - 180)

shoulder_tilt_error = axial_difference(user_angle, idol_angle)
hip_tilt_error = axial_difference(user_angle, idol_angle)
torso_lean_error = circular_difference(user_angle, idol_angle)

support_offset_x = ankle_center_x - hip_center_x
support_center_error =
  abs(user_support_offset_x - idol_support_offset_x)

stance_width = abs(right_ankle_x - left_ankle_x)
stance_width_error =
  abs(user_stance_width - idol_stance_width)
```

shoulder/hip/torso는 양쪽 pose의 해당 벡터가 모두 `epsilon`보다 큰 frame만
valid다. support/stance는 양쪽 ankle span이 모두 `epsilon`보다 큰 frame만
valid다.

```text
component names =
  (shoulder_tilt, hip_tilt, torso_lean, support_center, stance_width)
weights = (0.20, 0.20, 0.25, 0.20, 0.15)
scales = (2.5, 2.5, 2.0, 250.0, 200.0)

component_score(i) =
  clip(100 - mean_error(i) * scale(i), 0, 100)

effective_weight(i) = weight(i), if valid_count(i) > 0, else 0
balance_score =
  sum(component_score(i) * effective_weight(i))
  / sum(effective_weight(i))

weighted_deduction(i) =
  effective_weight(i) * (100 - component_score(i))
  / sum(effective_weight)
sum(weighted_deduction) = 100 - balance_score
```

invalid component는 score/mean error가 `None`, deduction이 0이다. 일부
invalid이면 valid weight만 재정규화한다. 모두 invalid이면 score 0,
`reliable=false`, `fallback_reason="all_balance_components_invalid"`,
effective weight 합 0이다. worst component는 weighted deduction 내림차순,
동률이면 정의 순서 우선으로 `top_k`개 선정한다.

`BalanceDiagnostics`는 component 이름/score/mean error/configured
weight/scale/effective weight/valid-frame count/weighted deduction, 다섯
per-frame error 목록, effective weight 합, frame count, top_k, worst
components, reliable, fallback reason, `camera_assumption="frontal_2d"`를
포함한다. `to_dict()`는 JSON 직렬화 가능하다.

scale과 weight는 실제 사람 평가로 학습된 값이 아닌 MVP 초기 calibration
값이다. Balance는 DCA 예측이 아닌 규칙 기반 점수다. Rhythm과 종합 점수
(`rule_score`, `final_score`)는 아직 미구현이다.

## 9. Rhythm

### 목적

timing lag를 보정한 뒤 사용자와 기준의 움직임 속도 패턴이 얼마나 유사한지 평가한다. 이 점수는 음악 BPM 또는 실제 beat를 직접 분석하지 않는다. **pose motion signal 기반 동작 리듬 유사도**다.

### 계산식

timing에서 정렬한 공통 구간의 motion signal을 비교한다.

```text
motion_correlation =
  corr(normalized_user_motion, normalized_idol_motion)

correlation_score =
  clip(
    (motion_correlation + 1) / 2 * 100,
    0,
    100
  )

normalized_motion_mae =
  mean(
    abs(
      normalized_user_motion - normalized_idol_motion
    )
  )

mae_score =
  clip(
    100 - normalized_motion_mae * rhythm_mae_scale,
    0,
    100
  )

rhythm_score =
    0.6 * correlation_score
  + 0.4 * mae_score
```

0.6과 0.4는 MVP 초기값이다. 표준편차가 거의 0인 정지 구간에서는 correlation fallback이 필요하다. 다시 말해 rhythm은 음악 beat 분석 점수가 아니라 pose motion signal 기반 동작 리듬 유사도다.

### Diagnostics

- `score`
- `motion_correlation`
- `normalized_motion_mae`
- `correlation_score`
- `mae_score`
- `correlation_weight`
- `mae_weight`
- `aligned_motion_length`
- `user_motion_std`
- `idol_motion_std`

### 숫자 예시

```text
correlation_score = 84
mae_score = 72

rhythm_score = 0.6 * 84 + 0.4 * 72
             = 50.4 + 28.8
             = 79.2
```

설명용 예시이며 실제 calibration 결과가 아니다.

### 점수가 낮아지는 대표 사례

- 전체 lag는 보정됐지만 빠르고 느린 구간의 변화 패턴이 다름
- 기준의 motion peak를 사용자가 약하거나 과도하게 표현함
- 중간 동작을 급하게 수행하거나 불필요하게 멈춤

### 11-7 확정 Rhythm 공식

11-7에서 `score_rhythm.py`와 `test_score_rhythm.py`를 구현했다. 입력은
timing 정렬 후 normalized `(T_aligned, 33, 3)` pose다. 재정규화, timing
재계산, lag 재탐색은 하지 않는다. 공통 helper를 다음처럼 재사용한다.

```text
user_motion = compute_motion_signal(user, aligned_config)
idol_motion = compute_motion_signal(idol, aligned_config)
motion signal shape = (T_aligned - 1,)

user_stats = normalize_signal(user_motion, aligned_config)
idol_stats = normalize_signal(idol_motion, aligned_config)
```

normalization은 `(signal - mean)/(std + epsilon)`이고 `std <= epsilon`이면
같은 길이의 0 signal과 constant flag를 반환한다.

```text
user_centered = normalized_user_motion - mean(normalized_user_motion)
idol_centered = normalized_idol_motion - mean(normalized_idol_motion)

denominator =
  sqrt(sum(user_centered**2) * sum(idol_centered**2))

motion_correlation =
  sum(user_centered * idol_centered) / denominator

correlation_score =
  clip((motion_correlation + 1) / 2 * 100, 0, 100)

per_motion_abs_error =
  abs(normalized_user_motion - normalized_idol_motion)
normalized_motion_mae = mean(per_motion_abs_error)

mae_score =
  clip(100 - normalized_motion_mae * config.rhythm_mae_scale, 0, 100)

rhythm_score =
    config.rhythm_correlation_weight * correlation_score
  + config.rhythm_mae_weight * mae_score
```

기본 correlation weight는 0.60, MAE weight는 0.40이며
`rhythm_mae_scale`은 config의 기존 값 25.0을 쓴다. 모두 MVP 초기
calibration 값이다. correlation은 부동소수점 오차를 고려해 `[-1,1]`로
clipping한다.

fallback은 다음과 같다.

- `both_constant`: correlation/score는 `None`, MAE 0, MAE score와 최종
  score는 100, unreliable
- `one_constant`: correlation/score는 `None`, 실제 normalized MAE와 MAE
  score를 diagnostics에 남기고 최종 score는 0, unreliable
- `correlation_unavailable`: 둘 다 nonconstant지만 denominator가
  `epsilon` 이하일 때 correlation weight를 0으로 보고 최종 score를
  `mae_score`로 반환, unreliable

`RhythmDiagnostics`는 correlation/MAE 및 점수, config weight/scale,
frame count와 aligned motion length, 양쪽 motion mean/std/constant flag,
원본 및 normalized signal, per-motion absolute errors, 신뢰도/fallback,
rhythm 정의와 audio/BPM 비분석 flag를 포함한다.

Rhythm은 pose motion pattern 유사도이며 음악 BPM, beat tracking 또는 audio
점수가 아니다. 표준화된 signal을 사용하므로 모든 motion 크기를 동일한 양수
배율로 바꾼 절대 intensity 차이는 제거될 수 있다. Rhythm은 DCA 예측이 아닌
규칙 기반 점수다. Timing, Accuracy, Detail, Balance, Rhythm의 개별 점수는
구현됐지만 `rule_score`, `final_score`, 통합 JSON은 아직 미구현이다.

## 10. Rule score 숫자 예시

```text
accuracy = 76
detail = 74
balance = 82
timing = 85
rhythm = 79.2

rule_score =
    0.30 * 76
  + 0.15 * 74
  + 0.20 * 82
  + 0.15 * 85
  + 0.20 * 79.2

= 22.80 + 11.10 + 16.40 + 12.75 + 15.84
= 78.89

final_score = 78.89
```

이 계산은 설명용이며 실제 사용자 영상이나 사람 평가를 통해 calibration한 결과가 아니다.

## 11. Diagnostics 반환 원칙

### 11-8 확정 종합 점수

`pentagon_scores.py`와 `test_pentagon_scores.py`를 구현했다.

```text
raw pose -> Timing(normalize + lag + align)
         -> Accuracy/Detail/Balance/Rhythm(aligned normalized pose)

rule_score =
    accuracy * 0.30
  + detail   * 0.15
  + balance  * 0.20
  + timing   * 0.15
  + rhythm   * 0.20

final_score = rule_score
score_mode = "rule_only"
pipeline_version = "pentagon_rule_v1"
```

Timing만 정규화와 lag 탐색을 수행해 전역 지연의 중복 감점을 막는다. optional
`dca_score`는 finite 0~100 passthrough 값이며 final score에는 사용하지 않는다.

각 component diagnostics에 reliability가 있으면 사용하고 없으면 true다.
overall은 다섯 값의 AND다. unreliable component도 반환 score를 고정 weight로
가중합하며 제외하거나 weight를 재정규화하지 않는다.

`component_results`는 다섯 `{score, diagnostics}`를 담는다.
`feedback_summary`는 timing lag, accuracy worst joint, detail worst
position/angle, balance worst component, rhythm correlation/MAE를 요약한다.
정렬 pose와 원본 pose 배열은 JSON에서 제외하고 모든 NumPy 값과 dataclass는
JSON-compatible Python 기본형으로 변환한다.

개별 오각형 점수와 종합 점수 구현은 완료되었다. 실제 영상 calibration은
여전히 필요하며 DCA 결합과 서비스/WebSocket 연동은 아직 미구현이다.

각 점수는 최종 score만 반환하지 않고 원시 오차, 적용 scale/weight, 유효 frame 수와 품질 신호를 diagnostics로 제공해야 한다. 개발 및 calibration에서는 전체 diagnostics를 항상 사용한다. 외부 API에서는 응답 크기와 내부 정보 노출을 고려하여 `include_diagnostics=true` 또는 debug 모드에서 전체를 반환하는 방식을 권장한다.

`per_joint_errors`, `joint_weights`, signal 통계와 선택 index는 결과를 설명하고 calibration 오류를 찾는 데 중요하다. 모든 scale, weight 및 penalty는 학습된 인간 평가 기준이 아니라 MVP 초기 calibration 상수라는 metadata 또는 문서 버전을 함께 관리해야 한다.

## 12. 구현 전 결정해야 할 사항

- lag 부호와 `lag_direction`의 정확한 계약
- 호출 측 실제 입력의 `[x, y, z]` 계약 준수 여부 및 `z_weight` calibration
- joint importance/visibility weight 결합 방식
- endpoint 목록, 발 중심 정의, angle 단위
- 정지 신호의 correlation fallback
- 최소 정렬 frame 수와 30프레임 window 경계 처리
- 모든 scale, penalty, lag 제한 및 epsilon
- 실제 사용자 영상과 사람이 직접 채점한 데이터에 기반한 calibration 방법

11-2에서는 위 정규화와 공통 수학 helper까지만 구현했다. 오각형 다섯 항목의 최종 점수는 아직 구현하지 않았다.
