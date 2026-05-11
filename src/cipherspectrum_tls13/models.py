from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (for adversarial debiasing)
# ---------------------------------------------------------------------------

class _GRLFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Multiplies gradient by -lambda during backprop, leaving forward unchanged."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.lambda_)


class CipherDiscriminator(nn.Module):
    """Lightweight MLP that predicts cipher suite from backbone features.
    Used together with GRL to encourage cipher-agnostic representations.
    """

    def __init__(self, in_dim: int, num_ciphers: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_ciphers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class ByteBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer baseline
# ---------------------------------------------------------------------------

class TrafficTransformerClassifier(nn.Module):
    def __init__(self, seq_dim: int, byte_dim: int, num_classes: int, d_model: int = 128, nhead: int = 8, layers: int = 3):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.byte_branch = ByteBranch(byte_dim, hidden_dim=d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_bytes: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        seq_h = self.seq_proj(x_seq)
        seq_h = self.encoder(seq_h)
        seq_h = seq_h.mean(dim=1)
        byte_h = self.byte_branch(x_bytes)
        features = torch.cat([seq_h, byte_h], dim=1)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits


# ---------------------------------------------------------------------------
# NetMambaLite (SSM-inspired with depthwise conv + GLU gate)
# ---------------------------------------------------------------------------

class MambaLiteBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # Depthwise conv captures local byte-level stride patterns
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        # GLU gate for selective state update
        self.gate = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GLU(dim=-1))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        conv_in = h.transpose(1, 2)
        conv_out = self.dwconv(conv_in).transpose(1, 2)
        h = self.gate(conv_out)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x


class NetMambaLiteClassifier(nn.Module):
    """NetMamba-inspired classifier with optional adversarial cipher debiasing."""

    def __init__(
        self,
        seq_dim: int,
        byte_dim: int,
        num_classes: int,
        d_model: int = 128,
        layers: int = 4,
        num_ciphers: int = 3,
        use_adversarial_debiasing: bool = False,
        adversarial_lambda: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(seq_dim, d_model)
        self.blocks = nn.ModuleList([MambaLiteBlock(d_model) for _ in range(layers)])
        self.byte_branch = ByteBranch(byte_dim, hidden_dim=d_model)
        feat_dim = d_model * 2
        self.head = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )
        self.use_adversarial_debiasing = use_adversarial_debiasing
        if use_adversarial_debiasing:
            self.grl = GradientReversalLayer(lambda_=adversarial_lambda)
            self.cipher_discriminator = CipherDiscriminator(feat_dim, num_ciphers=num_ciphers)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_bytes: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.in_proj(x_seq)
        for block in self.blocks:
            x = block(x)
        seq_h = x.mean(dim=1)
        byte_h = self.byte_branch(x_bytes)
        features = torch.cat([seq_h, byte_h], dim=1)
        logits = self.head(features)

        if self.use_adversarial_debiasing:
            cipher_logits = self.cipher_discriminator(self.grl(features))
            return logits, cipher_logits

        if return_features:
            return logits, features
        return logits


# ---------------------------------------------------------------------------
# 1D-CNN baseline
# ---------------------------------------------------------------------------

class Traffic1DCNNClassifier(nn.Module):
    """Lightweight 1D-CNN baseline for SOTA comparison.

    Three stacked Conv1d blocks with residual-style skip connections,
    followed by global average pooling.  A ByteBranch processes the
    payload byte histogram in parallel, mirroring the Transformer and
    Mamba branches.
    """

    def __init__(self, seq_dim: int, byte_dim: int, num_classes: int, d_model: int = 128):
        super().__init__()
        self.in_proj = nn.Linear(seq_dim, d_model)
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            # Block 2
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            # Block 3
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.byte_branch = ByteBranch(byte_dim, hidden_dim=d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_bytes: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x_seq: (B, T, seq_dim)
        h = self.in_proj(x_seq)          # (B, T, d_model)
        h = h.transpose(1, 2)            # (B, d_model, T) for Conv1d
        h = self.conv_blocks(h)
        seq_h = h.mean(dim=-1)           # global average pooling → (B, d_model)
        byte_h = self.byte_branch(x_bytes)
        features = torch.cat([seq_h, byte_h], dim=1)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits


# ---------------------------------------------------------------------------
# BiLSTM baseline
# ---------------------------------------------------------------------------

class TrafficLSTMClassifier(nn.Module):
    """Bidirectional LSTM baseline for SOTA comparison.

    Two-layer BiLSTM processes the packet sequence; the last hidden
    state of both directions is concatenated.  A ByteBranch handles
    the payload byte histogram, consistent with the other architectures.
    """

    def __init__(self, seq_dim: int, byte_dim: int, num_classes: int, d_model: int = 128, layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(seq_dim, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.byte_branch = ByteBranch(byte_dim, hidden_dim=d_model)
        # BiLSTM output dim = d_model * 2 (forward + backward last hidden)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2 + d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_bytes: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(x_seq)          # (B, T, d_model)
        _, (hn, _) = self.lstm(h)        # hn: (num_layers*2, B, d_model)
        # Grab the last layer's forward and backward hidden states
        seq_h = torch.cat([hn[-2], hn[-1]], dim=1)  # (B, d_model*2)
        byte_h = self.byte_branch(x_bytes)
        features = torch.cat([seq_h, byte_h], dim=1)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits


def create_model(
    model_name: str,
    seq_dim: int,
    byte_dim: int,
    num_classes: int,
    num_ciphers: int = 3,
    use_adversarial_debiasing: bool = False,
    adversarial_lambda: float = 0.1,
) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "transformer":
        return TrafficTransformerClassifier(seq_dim=seq_dim, byte_dim=byte_dim, num_classes=num_classes)
    if model_name == "mamba_lite":
        return NetMambaLiteClassifier(
            seq_dim=seq_dim,
            byte_dim=byte_dim,
            num_classes=num_classes,
            num_ciphers=num_ciphers,
            use_adversarial_debiasing=use_adversarial_debiasing,
            adversarial_lambda=adversarial_lambda,
        )
    if model_name == "cnn1d":
        return Traffic1DCNNClassifier(seq_dim=seq_dim, byte_dim=byte_dim, num_classes=num_classes)
    if model_name == "lstm":
        return TrafficLSTMClassifier(seq_dim=seq_dim, byte_dim=byte_dim, num_classes=num_classes)
    raise ValueError(f"Unknown model_name: {model_name}")
