# DCA-Net 규칙 기반 오각형 평가 설계 (11-1)

## 1. 문서 범위와 상태

이 문서는 현재 저장소 코드를 조사하고 규칙 기반 오각형 평가 파이프라인을 설계한 11-1 단계 산출물이며, 11-2 공통 기반, 11-3 timing, 11-4 accuracy 및 11-5 detail 구현을 반영해 갱신했다. 11-5에서는 `score_detail.py`와 `test_score_detail.py`를 구현했으며 balance, rhythm 및 종합 점수는 아직 구현하지 않았다. 서비스 코드와 기존 DCA 모델도 변경하지 않았다.

아래에서는 다음 표기를 사용한다.

- **확인된 사실**: 현재 Python 소스에서 직접 확인한 내용
- **설계안**: 이후 단계에서 구현할 정책 또는 인터페이스
- **확인 필요**: 현재 저장소만으로 확정할 수 없는 내용

일부 소스는 문자 인코딩이 깨진 주석을 포함한다. 또한 여러 실행문이 `#` 뒤 같은 줄에 있어 Python에서 주석으로 처리되는 상태다. 문서의 코드 조사는 실제 구문 상태를 기준으로 한다.

## 2. 현재 저장소 실제 구조

작업 전 확인한 구조는 다음과 같다.

```text
DCA-Net/
├── .DS_Store
├── generate_data.py
├── inference.py
├── model.py
└── train.py
```

- **확인된 사실**: `src` 또는 기타 하위 폴더가 없었다.
- **확인된 사실**: `.git` 폴더가 없고 Git 저장소가 아니다.
- **확인된 사실**: `service.py`, `normalization.py`, `dataset.py`, `realtime_buffer.py`, `config.py`가 없다.
- **확인된 사실**: WebSocket, 실시간 입력 버퍼, HTTP/API 서비스 구현이 없다.

## 3. 기존 파일 조사

### 3.1 `generate_data.py`

- 주요 역할: 기준 안무 배열을 복사하여 공간 노이즈와 선택적 프레임 shift를 적용한 synthetic 사용자 샘플 및 점수 label을 만들고 JSON 저장을 시도하는 스크립트
- 주요 함수: `generate_synthetic_user_data(idol_sequence, num_variants=100)`
- 함수 입력: 주석과 unpacking상 `idol_sequence`는 `(frames, joints, coords)`
- 예시 입력: `mock_idol_data = np.random.rand(30, 33, 3)`이므로 `(30, 33, 3)`
- 함수 출력: 길이 `num_variants`인 Python list. 각 원소는 `sample_id`, `score`, `user_dance`, `idol_dance`를 갖는 dict
- 실제 변수명:
  - 사용자 sequence: `user_dance`
  - 기준 sequence: `idol_sequence`, 저장 필드명 `idol_dance`
- pseudo/전체 점수 label: 의도된 식은 `mse = mean((idol_sequence - user_dance)^2)`, `score = max(0, 100 - mse * 200)`
- joint error label: 생성하지 않음
- 데이터 성격: 실제 사용자 데이터가 아니라 난수 기준 배열과 노이즈로 생성한 synthetic 데이터

중요한 현재 코드 상태:

- `error_factor = random.uniform(0.0, 0.5)`가 `#` 뒤에 있어 주석 처리되어 있다. 다음 줄에서 이를 사용하므로 함수를 실행하면 `NameError`가 발생할 수 있다.
- 선택적 shift의 `if random.random() > 0.7:`도 `#` 뒤에 있어 주석 처리되어 있다. 따라서 뒤의 들여쓰기 블록 때문에 구문 상태도 점검/수정이 필요하다.
- `mse = ...` 및 JSON 저장의 `with open(...)`도 `#` 뒤에 있어 주석 처리되어 있다.
- **확인 필요**: 깨진 주석의 원래 의도와 원본 인코딩
- **확인된 사실**: 이 파일은 정상적으로 동작하는 확정 데이터 생성기로 간주할 수 없다.

### 3.2 `model.py`

- 주요 역할: 관절 좌표를 공간·시간 Transformer로 인코딩하고 사용자와 기준 특징을 cross-attention으로 비교해 단일 점수를 출력하도록 의도된 DCA-Net 정의
- 주요 클래스:
  - `PositionalEncoding`
  - `DCANet`
- 주요 메서드:
  - `PositionalEncoding.forward(x)`
  - `DCANet.extract_features(x)`
  - `DCANet.forward(user_seq, idol_seq)`
- 생성자 기본값: `num_joints=33`, `in_channels=3`, `embed_dim=128`, `num_heads=4`
- 입력 shape: `extract_features`가 `B, T, V, C = x.shape`로 받고, 문서 문자열은 `(Batch, Frames, Joints, Coordinates)`를 명시한다. 기본 의도는 `(B, T, 33, 3)`.
- feature 출력:
  - 관절 projection: `(B, T, V, 128)`
  - spatial pooling 이후: `(B*T, 128)`
  - `extract_features` 반환: `(B, T, 128)`
- 모델 최종 출력: `forward`의 반환은 `score.squeeze(-1) * 100.0`, shape `(B,)`, 값 범위는 Sigmoid에 의해 의도상 `0~100`
- 모델은 joint별 error, highlight joint, 오각형 점수를 출력하지 않는다. 단일 batch score만 반환하도록 작성되어 있다.
- 실제 변수명: `user_seq`, `idol_seq`, `user_feat`, `idol_feat`

중요한 현재 코드 상태:

- `encoder_layer = nn.TransformerEncoderLayer(...)`의 시작이 `#` 뒤 같은 줄에 있어 주석 처리되어 있다. 이어지는 인자 줄 때문에 현재 소스는 정상 실행 가능한 모델 정의인지 재검토가 필요하며, `encoder_layer` 사용 전에 유효하게 생성되지 않는다.
- `score_head`라는 attribute는 `DCANet`에 없다. 점수 head의 실제 이름은 `fusion_mlp`이다.

### 3.3 `train.py`

- 주요 역할: 임시 dataset, DataLoader, Smooth L1 loss, Adam, cosine scheduler로 DCA-Net 학습을 수행하고 checkpoint를 저장하도록 의도된 스크립트
- 주요 클래스/함수:
  - `DanceDataset`
  - `DanceDataset.__len__`
  - `DanceDataset.__getitem__`
  - `train_model()`
- dataset 기본 shape:
  - `self.user_data`: `(1000, 30, 33, 3)`
  - `self.idol_data`: `(1000, 30, 33, 3)`
  - `self.labels`: `(1000, 1)`
  - 단일 item: user `(30, 33, 3)`, idol `(30, 33, 3)`, label `(1,)`
  - 기본 batch size 32일 때 user/idol `(32, 30, 33, 3)`, target `(32, 1)`
- 실제 변수명: `user_data`, `idol_data`, batch 변수 `user_seq`, `idol_seq`
- label 생성: `torch.randint(40, 100, (num_samples, 1)).float()`. pose 차이와 무관한 임의 정수 label
- 데이터 성격: `torch.randn`으로 각각 독립 생성한 synthetic 사용자/기준 데이터
- joint error label: 없음
- 학습 loss: `SmoothL1Loss(pred_score, target_score.squeeze(-1))`
- checkpoint 저장 의도: `torch.save(model.state_dict(), "dca_net_weights.pth")`

중요한 현재 코드 상태:

- 모델 생성문 `model = DCANet().to(device)`가 `#` 뒤에 있어 주석 처리되어 다음의 `model` 사용이 실패한다.
- `pred_score = model(...)`, `loss = ...`, `torch.save(...)`도 각각 `#` 뒤에 있어 주석 처리되어 있다.
- **확인된 사실**: 현재 저장소에는 checkpoint 파일이 없다.
- **확인된 사실**: 실제 사용자 영상이나 사람이 채점한 label을 로드하지 않는다.

### 3.4 `inference.py`

- 주요 역할: CPU에서 기준 sequence feature를 한 번 계산한 뒤 임의 사용자 sequence와 cross-attention 비교를 수행하는 “실시간 추론 시뮬레이션” 의도
- 주요 함수: `real_time_inference_simulation()`
- 입력으로 생성하는 tensor:
  - `idol_seq = torch.randn(1, 30, 33, 3)`
  - `user_seq = torch.randn(1, 30, 33, 3)`
- 실제 변수명: 사용자 `user_seq`, 기준 `idol_seq`
- 중간 feature: `precomputed_idol_feat`, `user_feat`; 의도 shape `(1, 30, 128)`
- checkpoint 로드 방식: `model.load_state_dict(torch.load("dca_net_weights.pth"))`가 주석 처리되어 있어 실제로 불러오지 않는다. `map_location`, checkpoint metadata, version 검증도 없다.
- 실제 반환 구조: 함수에 `return`문이 없으므로 Python 반환값은 `None`이다. JSON/dict 반환 구조가 없다.
- 출력 동작: 콘솔에 단일 `final_score.item()`을 출력하려고 한다.
- `score_100`, `joint_errors`, `highlight_joints`: 실제 코드에 없음

중요한 현재 코드 상태:

- `model.score_head(global_diff)`를 호출하지만 `DCANet`에는 `score_head`가 없다.
- `model.forward`의 실제 `fusion_mlp` 경로와 다른 임의 추론 경로를 사용한다.
- 파일 마지막에서 함수를 import guard 없이 즉시 실행한다.
- 따라서 현재 코드만으로 정상적인 실시간 서비스나 안정적인 반환 API를 제공하지 못한다.

## 4. 입력 좌표, 길이 및 전처리에 관한 확인 결과

| 조사 항목 | 현재 코드에서 확인한 결과 |
|---|---|
| sequence 길이 | 예시와 dataset 기본값은 30프레임. 모델 자체는 `T`를 가변으로 받으므로 항상 30으로 강제하지 않음 |
| 관절 수 | 생성/학습/추론 예시는 33, `DCANet` 기본값도 33. 입력 검증으로 강제하지는 않음 |
| 좌표 차원 | 생성/학습/추론 예시 및 `in_channels` 기본값은 3 |
| 세 번째 좌표의 의미 | 기존 DCA 코드는 3채널일 뿐 실제 z인지 확인되지 않음. 새 오각형 모듈은 호출 측이 `[x, y, z]`를 보장하는 명시적 계약을 채택 |
| 정규화 로직 | 기존 DCA 코드에는 없음. 11-2 `pentagon_common.py`에는 골반 중심 이동과 sequence 중앙 어깨너비 scaling 구현 |
| 골반 중심 정렬 | 기존 DCA 코드에는 없음. 새 오각형 모듈은 매 프레임 landmarks 23/24의 중점을 차감 |
| 어깨너비 scaling | 기존 DCA 코드에는 없음. 새 오각형 모듈은 landmarks 11/12의 XY 거리 중 유효값 중앙값 하나로 전체 sequence를 scaling |
| 좌우/카메라 좌표 처리 | 없음 |
| dtype 검증 | 없음 |
| shape 검증 | 명시적 검증 없음 |
| NaN/Inf 검증 | 없음 |

따라서 `(30, 33, 3)`이라는 shape만으로 세 번째 값을 실제 z로 단정하지 않는다. 새 오각형 모듈의 계약은 `[x, y, z]`이며, **호출 측이 실제 입력의 의미를 보장해야 한다**. 이는 기존 DCA prototype이 실제 MediaPipe z로 학습됐다는 증거가 아니다. 실제 pose 추출기, 좌표계, visibility 포함 여부와 축 방향은 여전히 확인해야 한다.

## 5. 확인된 사실과 확인 필요 사항

### 확인된 사실

- 현재 데이터는 synthetic 난수 데이터이며 실제 사용자 영상 데이터가 아니다.
- `generate_data.py`의 의도된 pseudo score는 전체 좌표 MSE를 `100 - 200*MSE`로 매핑해 0 아래를 자르는 방식이다.
- `train.py`의 label은 pose 기반 pseudo label조차 아니며 40 이상 100 미만의 무작위 정수다.
- 전체 score label 하나만 있고 joint error label은 없다.
- DCA-Net의 의도된 실제 `forward` 출력은 batch별 단일 0~100 score `(B,)`이다.
- inference 함수는 dict/JSON을 반환하지 않고 `None`을 반환한다.
- `score_100`, `joint_errors`, `highlight_joints`라는 필드명은 기존 코드에 없다.
- checkpoint 로드 코드는 주석 상태이고 checkpoint 파일도 현재 구조에 없다.
- 기존 DCA 파일에는 좌표 정규화, 골반 중심 정렬, 어깨너비 scaling이 구현되어 있지 않다.
- 11-2 새 오각형 공통 모듈에는 입력 검증, 골반 중심 정렬, sequence 중앙 어깨너비 scaling 및 공통 수학 helper가 구현되었다.
- service, WebSocket, realtime buffer 코드는 없다.

### 확인 필요

- 실제 MediaPipe 추출 파이프라인과 세 번째 좌표가 실제 z인지 여부
- 실제 입력 좌표의 단위, 축 방향, 좌우 반전 정책 및 visibility/confidence 사용 여부
- 사용자와 기준 영상의 실제 FPS 및 30프레임 window 생성 방식
- 정면 촬영을 제품 요구사항으로 강제할지 여부
- 실제 checkpoint의 존재 위치, 학습 이력, 성능 및 호환성
- 깨진 소스 주석의 원래 인코딩과 의도
- lag 부호가 “사용자 지연/선행” 중 어느 방향을 뜻하는지
- 결측 landmark 및 낮은 confidence 관절 처리 정책
- 실제 사용자 영상과 사람 평가 데이터에 기반한 모든 scale/weight calibration

## 6. 오각형 평가 정책

### 점수 정의

1. `timing`: 기준 안무 대비 사용자의 전체 프레임 지연 또는 선행
2. `accuracy`: timing 정렬 후 전신 관절 위치 차이
3. `detail`: 손목·팔꿈치·발목·발끝 등 말단 위치와 팔꿈치·무릎 각도 차이
4. `balance`: 기준 대비 어깨 기울기, 골반 기울기, 몸 중심과 발 중심 차이
5. `rhythm`: timing 정렬 후 움직임 속도 패턴의 유사도

`rhythm`은 음악 BPM 또는 beat를 직접 분석하는 점수가 아니다. pose motion signal 기반의 **동작 리듬 유사도**다.

### 종합 정책

- `pentagon_scores`: 100% 규칙 기반
- `rule_score`: 오각형 다섯 점수의 가중합
- `final_score = rule_score`
- DCA-Net 단일 점수는 `dca_score`라는 별도 보조 점수
- 현재 `rule_score`와 `dca_score`는 혼합하지 않음
- `score_mode = "rule_based"`
- DCA checkpoint/추론을 사용할 수 없으면 `dca_score: null` 허용

MVP 초기 가중치는 다음과 같다.

| 항목 | 가중치 |
|---|---:|
| accuracy | 0.30 |
| detail | 0.15 |
| balance | 0.20 |
| timing | 0.15 |
| rhythm | 0.20 |

합은 1.00이다. 이는 확정 정답이 아니라 MVP 초기값이며, 실제 사용자 영상과 사람이 채점한 평가를 이용해 calibration해야 한다.

## 7. 전체 데이터 흐름

1. `user_seq`와 `idol_seq` 입력
2. shape 검사
3. dtype 검사
4. NaN과 Inf 검사
5. 두 sequence의 정규화 방식 일치 여부 확인
6. 프레임별 움직임량 신호 계산
7. 제한된 lag 범위에서 cross-correlation 계산
8. 가장 적절한 timing lag 탐색
9. timing score 계산
10. lag를 이용해 `user_seq`와 `idol_seq` 정렬
11. 정렬된 공통 프레임 구간 추출
12. accuracy 계산
13. detail 계산
14. balance 계산
15. rhythm 계산
16. 오각형 다섯 점수의 가중합으로 `rule_score` 계산
17. `final_score = rule_score` 설정
18. 기존 DCA-Net 점수를 `dca_score`로 별도 반환
19. diagnostics를 포함한 JSON 생성

timing을 먼저 계산하는 핵심 이유는 사용자가 기준보다 몇 프레임 늦거나 빠르다는 단일 원인 때문에 accuracy, detail, balance, rhythm까지 모두 중복 감점되는 현상을 줄이기 위해서다. lag 자체는 timing에서 평가하고, 나머지 항목은 공통 정렬 구간에서 동작의 질을 평가한다.

### 7.1 11-3 timing 계약

`compute_motion_signal`로 정규화된 pose의 연속 frame 관절 변위 가중 평균을 만들고, 각 signal을 표준화한 뒤 `[-max_lag_frames, +max_lag_frames]`를 검색한다. 각 lag에서는 잘린 실제 overlap 구간의 평균을 다시 제거해 Pearson correlation을 계산한다. 후보 순서는 `0, -1, 1, -2, 2, ...`이며 correlation 차이가 `epsilon` 이하인 tie에서는 이 순서를 유지한다. 따라서 0, 더 작은 절댓값, 같은 절댓값의 음수 순으로 결정된다.

- `lag_frames == 0`: `aligned`
- `lag_frames > 0`: 사용자가 늦은 `user_delayed`; `user[lag:]`와 `idol[:-lag]`
- `lag_frames < 0`: 사용자가 빠른 `user_ahead`; `user[:lag]`와 `idol[-lag:]`

motion overlap은 최소 `minimum_aligned_frames - 1`개여야 한다. 정렬된 pose 공통 구간은 `T - abs(lag_frames)` frame이며, 이후 accuracy/detail/balance/rhythm이 이 동일 구간을 사용할 예정이다.

두 motion signal이 모두 constant면 timing 위반을 관찰할 수 없으므로 lag 0, score 100, `reliable=false`, `fallback_reason="both_constant"`를 반환한다. 한쪽만 constant면 대응을 만들 수 없으므로 lag 0, score 0, `reliable=false`, `fallback_reason="one_constant"`를 반환한다. 정상 검색의 `reliable`은 최고 correlation이 0보다 큰지를 뜻한다. 낮은 correlation에서도 선택 lag가 0이면 lag 기반 점수는 높을 수 있으며, 이는 동작 유사성의 증명이 아니라 전역 지연 감점이 작다는 뜻이다.

timing은 음악 beat 분석이 아니라 기준 pose motion signal과 사용자 pose motion signal의 전역 프레임 지연을 평가한다. `max_lag_frames`와 `timing_penalty_per_frame`는 MVP calibration 값이다. 기본 30프레임 window는 검색 가능한 지연과 관찰 구간이 짧고 긴 안무·경계 동작을 충분히 표현하지 못한다.

### 7.2 11-4 accuracy 계약

accuracy 입력은 timing에서 반환한 정렬·정규화 완료 pose 두 개이며 shape는 `(T_aligned, 33, 3)`이다. `T_aligned = T - abs(lag_frames)`이므로 30이 아닐 수 있다. 구현은 원본 `PentagonConfig`를 바꾸지 않고 `replace(config, expected_frames=None)`로 검증하며, `minimum_aligned_frames` 이상인 동일 shape만 허용한다. accuracy 내부에서는 pose 정규화나 timing 추정을 다시 수행하지 않는다.

33개 관절 모두의 좌표 거리를 공통 `compute_weighted_joint_distances`로 계산한다. `use_z=true`이면 `sqrt(dx² + dy² + z_weight·dz²)`, false이면 `sqrt(dx² + dy²)`이다. 관절별 frame 평균, 전체 단순 평균, 관절 중요도를 적용한 가중 평균, frame별 가중 평균 및 관절별 `weight × mean_error / weight_sum` 감점 기여도를 diagnostics에 기록한다.

`worst_joint_indices`는 좌표 오차 자체가 아니라 weighted contribution 내림차순이며, 동률은 낮은 MediaPipe index가 먼저다. 얼굴 landmark는 춤의 큰 자세와 팔다리 형태에 비해 낮은 가중치를, 몸통과 주요 팔다리는 더 높은 가중치를 갖는다. 이는 중요도에 대한 MVP 가정이며 실제 평가 데이터로 조정해야 한다.

accuracy는 timing 정렬 뒤 계산하여 전역 지연 하나가 timing과 관절 위치에서 중복 감점되는 것을 줄인다. 이 점수는 실제 음악 beat나 DCA 모델과 무관한 규칙 기반 점수다. `accuracy_scale=300`과 관절 가중치는 이론적으로 확정된 값이 아니라 MVP 점수 범위 조정을 위한 초기 calibration 값이며, 실제 사용자 영상과 사람 평가로 조정해야 한다.

### 7.3 11-5 detail 계약

11-5 산출물은 프로젝트 루트의 `score_detail.py`와
`test_score_detail.py`다. 구현 API는 `DetailDiagnostics`,
`DetailScoreResult`, `compute_detail_score(...)`이며, 두 dataclass의
`to_dict()`는 NumPy 객체를 남기지 않는 JSON 호환 Python 값만 반환한다.

detail 입력은 timing 정렬과 pose 정규화가 끝난 `(T_aligned, 33, 3)` 두 배열이다. `T_aligned`는 30이 아닐 수 있으므로 원본 config를 변경하지 않고 `replace(config, expected_frames=None)`로 검증한다. 최소 정렬 frame 수와 동일 shape를 요구하며 내부에서 pose 정규화, timing 계산 또는 accuracy 계산을 반복하지 않는다.

위치는 14개 말단 관절인 wrists 15/16, pinkies 17/18, indices 19/20, thumbs 21/22, ankles 27/28, heels 29/30, foot indices 31/32를 평가한다. 가중치는 순서대로 `(1.2, 1.2, 0.7, 0.7, 0.8, 0.8, 0.7, 0.7, 1.2, 1.2, 0.9, 0.9, 1.0, 1.0)`이다. 공통 거리 함수에 따라 `use_z=true`이면 `sqrt(dx²+dy²+z_weight·dz²)`, false이면 xy 거리만 사용한다.

각도는 left elbow `(11,13,15)`, right elbow `(12,14,16)`, left knee `(23,25,27)`, right knee `(24,26,28)`의 내각을 degree로 평가한다. 사용자와 기준 valid mask가 모두 true인 frame만 사용한다. 일부 각도가 전부 invalid면 해당 각도의 effective weight만 0으로 두고 나머지 각도로 계산한다. 네 각도가 모두 invalid면 `angle_score=None`, `reliable=false`, `fallback_reason="all_angles_invalid"`로 기록하고 position score를 100% 사용한다.

위치와 각도 각각의 `weight × mean_error / effective_weight_sum`을 감점 기여도로 기록한다. worst position은 contribution 내림차순, 동률이면 작은 MediaPipe index 순이다. worst angle은 contribution 내림차순, 동률이면 angle spec 정의 순서이며 최대 `min(top_k, 4)`개를 반환한다.

detail은 전역 지연을 중복 감점하지 않도록 timing 정렬 후 계산하는 규칙 기반 점수이며 DCA 모델 및 실제 음악 beat와 무관하다. position scale 350, angle scale 1.5, 위치·각도 가중치와 60:40 결합비는 모두 MVP 초기 calibration 값으로 실제 사용자 영상과 사람 평가로 조정해야 한다.

## 8. 전처리와 촬영 가정

- 초기 MVP는 정면 촬영을 기본 가정으로 검토한다.
- 카메라 회전, 원근, 좌우 반전, 신체 비율 차이가 점수에 큰 영향을 줄 수 있으므로 촬영 가정을 API와 UI에 명시해야 한다.
- 사용자와 기준 sequence는 동일한 landmark 정의 및 동일한 좌표 정규화 정책을 사용해야 한다.
- 새 오각형 모듈의 입력 계약은 NumPy 호환 `(T, 33, 3)`의 `[x, y, z]`, 계산 dtype `float32`, finite 값이다. 기본 `T=30`은 설정으로 변경할 수 있다. shape만으로 좌표 의미를 추론하지 않는다.
- 각 프레임에서 왼쪽/오른쪽 골반(23/24)의 중점을 모든 관절에서 차감한다.
- 왼쪽/오른쪽 어깨(11/12)의 x/y 평면 거리를 계산하고, 유효한 전체 sequence 어깨너비의 중앙값 하나를 scale로 사용한다. 기본 30프레임 입력이면 유효한 30개 값의 중앙값을 사용한다.
- 프레임별 scale을 사용하지 않는 이유는 landmark 검출 노이즈가 매 프레임 크기 jitter로 증폭되는 것을 줄이기 위해서다.
- 사용자와 기준은 체형 및 촬영 크기 차이를 줄이기 위해 각자의 중앙 어깨너비로 독립 정규화한다.
- x/y/z 모두 골반 중심을 빼고 같은 scale로 나눈다. 이후 거리 계산에서만 `use_z`와 `z_weight`로 z 기여도를 조절한다.
- 어깨너비 계산에는 카메라 기반 깊이의 불안정 가능성 때문에 z를 사용하지 않는다.
- 좌우 반전, 회전, 원근 및 카메라 자세 보정은 아직 수행하지 않는다.
- 기존 DCA prototype에는 이 정규화가 적용되지 않았으며, 실제 z로 학습됐다는 증거도 아니다.
- 30프레임 window는 짧은 동작만 표현하며 긴 안무, 긴 지연, 구간 경계에서 제한이 있다. 모델은 가변 `T`를 받을 수 있게 작성되었지만 현재 예시·dataset·추론은 모두 30프레임이다.
- 모든 scale과 weight는 MVP 후보이며 실제 사용자 영상과 사람 평가로 calibration해야 한다.

## 9. 향후 flat 파일 구조 제안

현재 저장소에 `src`가 없으므로 초기 구현은 프로젝트 루트의 flat 구조를 기본안으로 한다. 아래 파일은 이번 단계에 생성하지 않았다.

| 제안 파일 | 역할 |
|---|---|
| `pentagon_config.py` | **11-2 구현됨**: MediaPipe index, 실제 계산에 쓰는 `z_weight`, scale, rhythm 가중치, lag 제한, 프레임당 감점 및 설정 검증 |
| `pentagon_common.py` | **11-2 구현됨**: 입력 검증, 골반/어깨 기반 정규화, clipping, weighted distance, 관절 각도+유효 mask, motion signal, signal 정규화 |
| `score_timing.py` | **11-3 구현됨**: overlap별 Pearson correlation, signed lag, fallback, 점수, 정렬 및 diagnostics |
| `score_accuracy.py` | **11-4 구현됨**: 정렬된 33관절 거리, 가중 오차·기여도, 점수 및 diagnostics |
| `score_detail.py` | **11-5 구현됨**: 14개 말단 위치, 4개 팔꿈치·무릎 각도, invalid fallback, 점수 및 diagnostics |
| `score_balance.py` | 정면 2D의 5개 Balance component, 점수/diagnostics |
| `score_rhythm.py` | **11-7 구현됨**: 정렬 motion correlation/MAE, fallback, 점수/diagnostics |
| `pentagon_scores.py` | **11-8 구현됨**: 실행 순서, `rule_score`, JSON-compatible response |
| `test_score_timing.py` | **11-3 구현됨**: deterministic pose 기반 timing 단위 테스트 |
| `test_score_accuracy.py` | **11-4 구현됨**: 거리·가중치·정렬 연계 기반 accuracy 단위 테스트 |
| `test_score_detail.py` | **11-5 구현됨**: 위치·각도·valid mask·timing 연계 기반 detail 단위 테스트 |
| `test_score_rhythm.py` | **11-7 구현됨**: motion pattern·fallback·timing 연계 기반 rhythm 단위 테스트 |
| `test_pentagon_scores.py` | **11-8 구현됨**: 가중합·reliability·JSON·중복 감점 방지 테스트 |

## 10. 기존 `inference.py` 연동 후보

현재 `inference.py`에는 반환 구조가 없고 `score_head` 참조도 유효하지 않으므로, 즉시 그 함수 안에 점수 계산을 삽입하는 방식은 권장하지 않는다. 11-2 이후 다음 경계를 제안한다.

1. `pentagon_scores.py`가 정규화 정책이 확인된 `user_seq`, `idol_seq`, `fps`를 받아 규칙 점수 response를 생성한다.
2. DCA 추론은 별도 adapter/helper에서 `model(user_seq, idol_seq)`를 호출하여 단일 값을 얻는다.
3. 성공하면 그 값을 `dca_score`에 넣고, checkpoint 부재·호환 오류·비활성 설정이면 `null`을 넣는다.
4. `final_score`는 DCA 성공 여부와 관계없이 `rule_score`와 동일하게 유지한다.
5. 기존 시뮬레이션은 후속 단계에서 명시적 요청이 있을 때만 고치며, 이번 단계에는 수정하지 않는다.

기존 코드에 `final_score`라는 지역 변수는 있지만 JSON field가 아니며, 현재 설계의 규칙 기반 `final_score`와 의미가 다르다. 통합 시 DCA 결과를 반드시 `dca_score`로 이름을 바꾸어 의미 충돌을 피한다. `highlight_joints`는 기존 필드가 아니라 새 schema 필드이며, accuracy/detail diagnostics에서 큰 오차 관절을 선정하는 규칙이 구현된 뒤에만 채운다.

## 11. 최종 JSON schema 기본안

```json
{
  "final_score": 78.89,
  "rule_score": 78.89,
  "dca_score": 81.2,
  "score_mode": "rule_based",
  "pentagon_scores": {
    "timing": 85.0,
    "accuracy": 76.0,
    "detail": 74.0,
    "balance": 82.0,
    "rhythm": 79.2
  },
  "diagnostics": {
    "timing": {
      "lag_frames": 3,
      "lag_ms": 100.0,
      "fps": 30,
      "penalty_per_frame": 5.0,
      "correlation_at_best_lag": 0.88
    },
    "accuracy": {
      "mean_joint_error": 0.07,
      "weighted_mean_joint_error": 0.08,
      "accuracy_scale": 300.0,
      "use_z": true,
      "z_weight": 0.3,
      "frame_count": 27,
      "worst_joint_indices": [15, 16]
    },
    "detail": {
      "position_score": 79.0,
      "angle_score": 82.0,
      "weighted_position_error": 0.06,
      "weighted_angle_error_deg": 12.0,
      "position_component_weight": 0.6,
      "angle_component_weight": 0.4,
      "reliable": true
    },
    "balance": {
      "shoulder_tilt_error": 0.04,
      "hip_tilt_error": 0.03,
      "centerline_error": 0.05
    },
    "rhythm": {
      "motion_correlation": 0.68,
      "normalized_motion_mae": 0.11,
      "correlation_score": 84.0,
      "mae_score": 72.0
    }
  },
  "highlight_joints": [11, 13, 15]
}
```

`dca_score`를 계산할 수 없으면 다음처럼 `null`을 허용한다.

```json
{
  "dca_score": null,
  "score_mode": "rule_based"
}
```

diagnostics는 내부 검증과 calibration에 필수지만 응답 크기가 크고 `per_joint_errors` 등이 포함될 수 있다. 따라서 개발 환경에서는 기본 활성화하고, 외부 API 기본값은 `include_diagnostics=false`로 두며 요청 시 전체 반환하는 방식을 권장한다. 최소 운영 응답에도 재현에 필요한 `timing.lag_frames`, `aligned_frame_count`, 설정 버전 또는 calibration 버전은 별도 요약 metadata로 남기는 방안을 검토한다.

## 12. Calibration과 설명 가능성

모든 scale, weight, lag 제한, penalty 및 clipping 상수는 학습된 인간 평가 기준이 아니라 MVP 초기 calibration 상수다. 각 점수는 최종 숫자뿐 아니라 원시 오차, 사용한 상수, 정렬 길이, 가장 나쁜 관절 등을 diagnostics로 반환하도록 설계한다. 실제 사용자 영상과 여러 평가자의 채점 데이터를 모아 분포, 상관관계, 편향, 촬영 조건별 안정성을 검증한 뒤 상수를 조정해야 한다.

## 13. 11-2 상태와 이후 구현 순서

11-2 완료 범위:

1. `[x, y, z]` 입력 좌표 계약 및 정규화 정책 확정
2. `pentagon_config.py`
3. `pentagon_common.py`
4. `test_pentagon_common.py` 공통 기반 테스트

11-3 이후 후보:

1. `score_timing.py`와 lag 부호 계약
2. `score_accuracy.py`
3. `score_detail.py`
4. `score_balance.py` (11-6 구현 완료)
5. `score_rhythm.py` (11-7 구현 완료)
6. `pentagon_scores.py` 통합 및 JSON schema (11-8 구현 완료)
7. `test_pentagon_scores.py`의 동일 sequence, lag, 노이즈, NaN/shape, 범위, schema 테스트
8. DCA checkpoint adapter와 `dca_score: null` fallback
9. 실제 사용자/사람 평가 데이터로 calibration
10. 별도 후속 단계에서 service, WebSocket, realtime buffer 설계 및 구현

현재 다섯 개별 점수와 `rule_score`, `final_score`, 통합 JSON-compatible
결과는 구현되었다. DCA 결합, 서비스 연동과 실제 영상 calibration은 남아 있다.

### 11-6 Balance 구현

`score_balance.py`와 `test_score_balance.py`를 추가했다. public API
`compute_balance_score(...)`는 timing 정렬 후 normalized
`(T_aligned, 33, 3)` pose를 입력받는다. 원본 config를 변경하지 않고
`expected_frames=None`인 복사본으로 가변 frame 수를 검증하며 원본의
`minimum_aligned_frames`를 적용한다. 재정규화와 timing 재계산은 하지 않는다.

Balance는 정면 2D 카메라의 x, y만 사용하는 규칙 기반 점수이며 z와 DCA
예측값을 사용하지 않는다. component는 `shoulder_tilt`, `hip_tilt`,
`torso_lean`, `support_center`, `stance_width`다. weight
`(0.20, 0.20, 0.25, 0.20, 0.15)`와 scale
`(2.5, 2.5, 2.0, 250.0, 200.0)`은 MVP 초기 calibration 값이다.

valid frame이 없는 component는 score/mean error가 `None`, effective weight가
0이며 나머지 weight를 재정규화한다. 모두 invalid이면 score 0,
`reliable=false`, `fallback_reason="all_balance_components_invalid"`이다.
diagnostics는 component별 score/error/weight/scale/valid count, per-frame
error, weighted deduction, worst component, 신뢰도와
`camera_assumption="frontal_2d"`를 제공한다. worst component는 weighted
deduction 내림차순, 동률이면 정의 순서로 선정한다.

### 11-7 Rhythm 구현

`score_rhythm.py`와 `test_score_rhythm.py`를 추가했다. public API
`compute_rhythm_score(...)`는 timing 정렬 후 normalized pose를 입력받고
재정규화, timing 재계산 또는 lag 재탐색을 하지 않는다. 공통
`compute_motion_signal`과 `normalize_signal`을 재사용하며 motion signal
길이는 `T_aligned - 1`이다.

두 normalized motion signal의 명시적 denominator 검증 Pearson correlation과
frame별 절대 차이의 평균인 normalized MAE를 사용한다. correlation score는
`(correlation+1)/2*100`, MAE score는
`100 - normalized_motion_mae*rhythm_mae_scale`을 clipping한다. 정상 최종
점수는 config의 correlation weight 0.60과 MAE weight 0.40을 적용한다.

둘 다 constant이면 관측 불가지만 mismatch도 없어 score 100
(`both_constant`), 한쪽만 constant이면 mismatch score 0
(`one_constant`)이다. 둘 다 nonconstant지만 correlation denominator가
`epsilon` 이하이면 MAE만 100% 사용하는 `correlation_unavailable`
fallback이다.

diagnostics는 원본/normalized motion signal, signal mean/std/constant flag,
correlation/MAE와 각 component 점수·설정, per-motion error, frame 및 motion
길이, 신뢰도와 fallback을 제공한다. 정의는
`pose_motion_similarity_after_timing_alignment`이며 audio와 BPM 분석 flag는
모두 false다. 표준화 때문에 동일한 양수 배율의 절대 motion intensity 차이는
제거될 수 있다. weight와 scale은 MVP 초기 calibration 값이며 Rhythm은 DCA
예측이 아닌 규칙 기반 점수다.

### 11-8 종합 점수 통합

`pentagon_scores.py`와 `test_pentagon_scores.py`를 구현했다. 원본 pose를
Timing에 전달해 정규화와 전역 lag 탐색을 한 번만 수행하고, 반환된 정렬
normalized pose를 Accuracy, Detail, Balance, Rhythm에 전달한다.

가중치는 Accuracy 0.30, Detail 0.15, Balance 0.20, Timing 0.15,
Rhythm 0.20이다. `rule_score`는 고정 가중합이며 현재
`final_score=rule_score`, `score_mode="rule_only"`다. optional `dca_score`는
검증 후 passthrough하지만 final score에는 사용하지 않는다.

reliability는 diagnostics의 `reliable`을 사용하고 없는 component는 true다.
overall은 논리 AND이며 unreliable 목록은 실행 순서로 기록한다. unreliable
component의 반환 점수도 고정 가중합에 사용하고 weight를 재정규화하지 않는다.

`component_results`는 다섯 score와 diagnostics를, `feedback_summary`는 lag,
worst joint/position/angle/component 및 rhythm correlation/MAE를 요약한다.
정렬·원본 pose 배열은 JSON에서 제외하며 NumPy 값과 dataclass를 Python
기본형으로 재귀 변환한다. pipeline version은 `pentagon_rule_v1`이다.
