import torch
import torch.nn as nn
import torch.nn.functional as F


FRAME_LENGTH = 30
NUM_JOINTS = 33
COORD_DIM = 3

# 현재 synthetic sample 수가 작기 때문에 큰 모델보다 작은 모델이 더 안정적일 수 있다.
# 그래서 7.6부터 기본 embed_dim=64, spatial/temporal layer=1로 낮춘다.
DEFAULT_EMBED_DIM = 64
DEFAULT_NUM_HEADS = 4
DEFAULT_SPATIAL_LAYERS = 1
DEFAULT_TEMPORAL_LAYERS = 1
DEFAULT_DROPOUT = 0.1
TOP_K_HIGHLIGHT = 3


def validate_dca_inputs(idol_seq, user_seq):
    if not torch.is_tensor(idol_seq):
        raise ValueError("idol_seq는 torch.Tensor여야 합니다.")

    if not torch.is_tensor(user_seq):
        raise ValueError("user_seq는 torch.Tensor여야 합니다.")

    if idol_seq.shape != user_seq.shape:
        raise ValueError(
            f"idol_seq와 user_seq shape가 같아야 합니다. "
            f"idol={tuple(idol_seq.shape)}, user={tuple(user_seq.shape)}"
        )

    if idol_seq.ndim != 4:
        raise ValueError(
            f"입력은 4차원 (B, 30, 33, 3)이어야 합니다. 현재: {tuple(idol_seq.shape)}"
        )

    expected_shape = (FRAME_LENGTH, NUM_JOINTS, COORD_DIM)

    if tuple(idol_seq.shape[1:]) != expected_shape:
        raise ValueError(
            f"입력 shape는 (B, {FRAME_LENGTH}, {NUM_JOINTS}, {COORD_DIM})이어야 합니다. "
            f"현재: {tuple(idol_seq.shape)}"
        )

    if torch.isnan(idol_seq).any() or torch.isnan(user_seq).any():
        raise ValueError("입력에 NaN 값이 있습니다.")

    if torch.isinf(idol_seq).any() or torch.isinf(user_seq).any():
        raise ValueError("입력에 Inf 값이 있습니다.")


def compute_diff_summary(idol_seq, user_seq):
    """
    원본 좌표 차이를 7개 숫자로 요약한다.
    feature-level diff와 함께 score head에 들어가며, score 예측의 직접 단서로 사용된다.
    """
    diff = torch.abs(user_seq - idol_seq)

    diff_mean = diff.mean(dim=(1, 2, 3))
    diff_max = diff.amax(dim=(1, 2, 3))

    per_joint_diff = diff.mean(dim=(1, 3))
    joint_mean = per_joint_diff.mean(dim=1)
    joint_max = per_joint_diff.max(dim=1).values
    joint_std = per_joint_diff.std(dim=1, unbiased=False)

    temporal_diff = diff.mean(dim=(2, 3))
    temporal_mean = temporal_diff.mean(dim=1)
    temporal_max = temporal_diff.max(dim=1).values

    diff_summary = torch.stack(
        [
            diff_mean,
            diff_max,
            joint_mean,
            joint_max,
            joint_std,
            temporal_mean,
            temporal_max,
        ],
        dim=1,
    )

    return diff_summary


class JointEmbedding(nn.Module):
    def __init__(
        self,
        coord_dim=COORD_DIM,
        embed_dim=DEFAULT_EMBED_DIM,
        num_joints=NUM_JOINTS,
    ):
        super().__init__()

        self.coord_proj = nn.Linear(coord_dim, embed_dim)
        self.joint_embedding = nn.Parameter(torch.zeros(1, 1, num_joints, embed_dim))

        nn.init.normal_(self.joint_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        return self.coord_proj(x) + self.joint_embedding


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=DEFAULT_EMBED_DIM,
        num_heads=DEFAULT_NUM_HEADS,
        num_layers=DEFAULT_SPATIAL_LAYERS,
        dropout=DEFAULT_DROPOUT,
    ):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        batch_size, frame_length, num_joints, embed_dim = x.shape

        x_flat = x.reshape(batch_size * frame_length, num_joints, embed_dim)
        spatial_tokens = self.encoder(x_flat)

        spatial_tokens = spatial_tokens.reshape(
            batch_size,
            frame_length,
            num_joints,
            embed_dim,
        )

        frame_features = spatial_tokens.mean(dim=2)

        return spatial_tokens, frame_features


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        frame_length=FRAME_LENGTH,
        embed_dim=DEFAULT_EMBED_DIM,
        num_heads=DEFAULT_NUM_HEADS,
        num_layers=DEFAULT_TEMPORAL_LAYERS,
        dropout=DEFAULT_DROPOUT,
    ):
        super().__init__()

        self.temporal_embedding = nn.Parameter(torch.zeros(1, frame_length, embed_dim))
        nn.init.normal_(self.temporal_embedding, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        x = x + self.temporal_embedding[:, : x.shape[1], :]
        return self.encoder(x)


class ChoreographyCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim=DEFAULT_EMBED_DIM,
        num_heads=DEFAULT_NUM_HEADS,
        dropout=DEFAULT_DROPOUT,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.compare_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, user_features, idol_features):
        attended_idol, attention_weights = self.attention(
            query=user_features,
            key=idol_features,
            value=idol_features,
        )

        comparison_input = torch.cat(
            [
                user_features,
                attended_idol,
                torch.abs(user_features - attended_idol),
                user_features * attended_idol,
            ],
            dim=-1,
        )

        comparison_features = self.compare_mlp(comparison_input)

        return attended_idol, attention_weights, comparison_features


class FeatureDiffFusion(nn.Module):
    """
    Cross Attention 결과에 user/idol temporal feature의 직접 차이를 추가로 주입한다.

    목적:
    - 기존 diff_summary 7개 숫자보다 더 강한 비교 단서 제공
    - 모델이 평균 점수만 출력하는 문제 완화
    """

    def __init__(self, embed_dim=DEFAULT_EMBED_DIM, dropout=DEFAULT_DROPOUT):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim * 6, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, comparison_features, user_temporal, idol_temporal):
        temporal_diff = user_temporal - idol_temporal
        temporal_abs_diff = torch.abs(user_temporal - idol_temporal)
        temporal_mul = user_temporal * idol_temporal

        fusion_input = torch.cat(
            [
                comparison_features,
                user_temporal,
                idol_temporal,
                temporal_diff,
                temporal_abs_diff,
                temporal_mul,
            ],
            dim=-1,
        )

        fused_features = self.net(fusion_input)

        return fused_features


class FeedbackDecoder(nn.Module):
    def __init__(
        self,
        coord_dim=COORD_DIM,
        embed_dim=DEFAULT_EMBED_DIM,
        num_heads=DEFAULT_NUM_HEADS,
        num_layers=1,
        dropout=DEFAULT_DROPOUT,
    ):
        super().__init__()

        self.input_proj = nn.Linear(coord_dim * 2, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(embed_dim, 1)

    def forward(self, idol_seq, user_seq):
        diff = torch.abs(user_seq - idol_seq)

        per_joint_mean_diff = diff.mean(dim=1)
        per_joint_max_diff = diff.max(dim=1).values

        joint_input = torch.cat(
            [per_joint_mean_diff, per_joint_max_diff],
            dim=-1,
        )

        joint_features = self.input_proj(joint_input)
        joint_features = self.encoder(joint_features)

        pred_joint_errors = self.out_proj(joint_features).squeeze(-1)

        return F.softplus(pred_joint_errors)


class ScoreHead(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim=DEFAULT_EMBED_DIM,
        dropout=DEFAULT_DROPOUT,
    ):
        super().__init__()

        hidden_dim = max(embed_dim // 2, 1)

        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, score_input):
        return self.net(score_input).squeeze(-1)


class GroovoDCANet(nn.Module):
    def __init__(
        self,
        frame_length=FRAME_LENGTH,
        num_joints=NUM_JOINTS,
        coord_dim=COORD_DIM,
        embed_dim=DEFAULT_EMBED_DIM,
        num_heads=DEFAULT_NUM_HEADS,
        spatial_layers=DEFAULT_SPATIAL_LAYERS,
        temporal_layers=DEFAULT_TEMPORAL_LAYERS,
        dropout=DEFAULT_DROPOUT,
        top_k=TOP_K_HIGHLIGHT,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim({embed_dim})은 num_heads({num_heads})로 나누어 떨어져야 합니다."
            )

        self.frame_length = frame_length
        self.num_joints = num_joints
        self.coord_dim = coord_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.spatial_layers = spatial_layers
        self.temporal_layers = temporal_layers
        self.dropout = dropout
        self.top_k = top_k
        self.diff_summary_dim = 7

        self.joint_embedding = JointEmbedding(
            coord_dim=coord_dim,
            embed_dim=embed_dim,
            num_joints=num_joints,
        )

        self.spatial_encoder = SpatialEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=spatial_layers,
            dropout=dropout,
        )

        self.temporal_encoder = TemporalEncoder(
            frame_length=frame_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=temporal_layers,
            dropout=dropout,
        )

        self.cross_attention = ChoreographyCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.feature_diff_fusion = FeatureDiffFusion(
            embed_dim=embed_dim,
            dropout=dropout,
        )

        self.diff_summary_norm = nn.LayerNorm(self.diff_summary_dim)

        self.score_head = ScoreHead(
            input_dim=embed_dim + self.diff_summary_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        self.feedback_decoder = FeedbackDecoder(
            coord_dim=coord_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=1,
            dropout=dropout,
        )

    def forward(self, idol_seq, user_seq):
        validate_dca_inputs(idol_seq, user_seq)

        # 7.6 모델은 score branch 구조가 바뀌었으므로 기존 checkpoint와 호환되지 않는다.
        # 반드시 6단계를 다시 실행해서 새 checkpoint를 만들어야 한다.

        idol_joint = self.joint_embedding(idol_seq)
        user_joint = self.joint_embedding(user_seq)

        _, idol_frame_features = self.spatial_encoder(idol_joint)
        _, user_frame_features = self.spatial_encoder(user_joint)

        idol_temporal = self.temporal_encoder(idol_frame_features)
        user_temporal = self.temporal_encoder(user_frame_features)

        _, attention_weights, comparison_features = self.cross_attention(
            user_temporal,
            idol_temporal,
        )

        fused_features = self.feature_diff_fusion(
            comparison_features,
            user_temporal,
            idol_temporal,
        )

        fused_pooled = fused_features.mean(dim=1)

        diff_summary = compute_diff_summary(idol_seq, user_seq)
        diff_summary = self.diff_summary_norm(diff_summary)

        score_input = torch.cat(
            [fused_pooled, diff_summary],
            dim=-1,
        )

        score_norm = self.score_head(score_input)
        score_100 = score_norm * 100.0

        pred_joint_errors = self.feedback_decoder(idol_seq, user_seq)

        k = min(self.top_k, pred_joint_errors.shape[1])
        highlight_joints = torch.topk(pred_joint_errors, k=k, dim=1).indices

        return {
            "score_norm": score_norm,
            "score_100": score_100,
            "pred_joint_errors": pred_joint_errors,
            "highlight_joints": highlight_joints,
            "attention_weights": attention_weights,
        }


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def create_dca_model(**kwargs):
    return GroovoDCANet(**kwargs)


def run_dummy_forward_test(device=None, batch_size=4):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_dca_model().to(device)
    model.eval()

    idol_seq = torch.randn(
        batch_size,
        FRAME_LENGTH,
        NUM_JOINTS,
        COORD_DIM,
        device=device,
    )

    user_seq = torch.randn(
        batch_size,
        FRAME_LENGTH,
        NUM_JOINTS,
        COORD_DIM,
        device=device,
    )

    with torch.no_grad():
        outputs = model(idol_seq, user_seq)

    total_params, trainable_params = count_parameters(model)

    print("===== Dummy Forward Output =====")

    for key, value in outputs.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: type={type(value)}")

    print(
        "score_norm range:",
        float(outputs["score_norm"].min()),
        "~",
        float(outputs["score_norm"].max()),
    )

    print("total parameters:", total_params)
    print("trainable parameters:", trainable_params)

    assert outputs["score_norm"].shape == (batch_size,)
    assert outputs["score_100"].shape == (batch_size,)
    assert outputs["pred_joint_errors"].shape == (batch_size, NUM_JOINTS)
    assert outputs["highlight_joints"].shape == (batch_size, TOP_K_HIGHLIGHT)

    return model
