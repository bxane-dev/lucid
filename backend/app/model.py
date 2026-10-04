import math

import torch
from torch import nn


ARCHITECTURES = (
    "eegnet",
    "cnn1d",
    "cnn_lstm",
    "temporal_cnn",
    "transformer",
    "eeg_conformer",
)


class EEGNet(nn.Module):
    def __init__(
        self,
        channels: int,
        classes: int,
        *,
        f1: int = 8,
        depth_multiplier: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        f2 = f1 * depth_multiplier

        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, 63),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(f1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                f1,
                f2,
                kernel_size=(channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                f2,
                f2,
                kernel_size=(1, 15),
                padding="same",
                groups=f2,
                bias=False,
            ),
            nn.Conv2d(
                f2,
                f2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 8)),
        )
        self.classifier = nn.Linear(
            f2 * 8,
            classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "EEGNet expects "
                "[batch, channels, samples]"
            )

        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return self.classifier(
            x.flatten(1)
        )


class CNN1D(nn.Module):
    def __init__(
        self,
        channels: int,
        classes: int,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(
                channels,
                64,
                kernel_size=9,
                padding=4,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(
                64,
                128,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(
                128,
                128,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(
            128,
            classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(
            self.features(x).squeeze(-1)
        )


class CNNLSTM(nn.Module):
    def __init__(
        self,
        channels: int,
        classes: int,
    ) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(
                channels,
                64,
                kernel_size=9,
                padding=4,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(
                64,
                64,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(
            192,
            classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.frontend(x)
        x = x.transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        representation = torch.cat(
            [hidden[-2], hidden[-1]],
            dim=1,
        )
        return self.classifier(
            representation
        )


class TemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = dilation * 2
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.activation(
            x + self.block(x)
        )


class TemporalCNN(nn.Module):
    def __init__(
        self,
        channels: int,
        classes: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(
                channels,
                96,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(96),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            TemporalBlock(96, 1),
            TemporalBlock(96, 2),
            TemporalBlock(96, 4),
            TemporalBlock(96, 8),
        )
        self.pool = (
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(
            96,
            classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def _sinusoidal_position(
    length: int,
    dimension: int,
    *,
    device,
    dtype,
) -> torch.Tensor:
    position = torch.arange(
        length,
        device=device,
        dtype=dtype,
    ).unsqueeze(1)
    div = torch.exp(
        torch.arange(
            0,
            dimension,
            2,
            device=device,
            dtype=dtype,
        )
        * (
            -math.log(10000.0)
            / dimension
        )
    )

    encoding = torch.zeros(
        length,
        dimension,
        device=device,
        dtype=dtype,
    )
    encoding[:, 0::2] = torch.sin(
        position * div
    )
    encoding[:, 1::2] = torch.cos(
        position * div
    )
    return encoding


class EEGTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        classes: int,
        *,
        d_model: int = 96,
    ) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(
                channels,
                d_model,
                kernel_size=9,
                stride=2,
                padding=4,
                bias=False,
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=192,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=3,
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(
            d_model,
            classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(x).transpose(1, 2)
        x = x + _sinusoidal_position(
            x.shape[1],
            x.shape[2],
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.classifier(x)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
    ) -> None:
        super().__init__()
        self.attention_norm = (
            nn.LayerNorm(dimension)
        )
        self.attention = (
            nn.MultiheadAttention(
                embed_dim=dimension,
                num_heads=4,
                dropout=0.2,
                batch_first=True,
            )
        )
        self.conv_norm = (
            nn.LayerNorm(dimension)
        )
        self.depthwise = nn.Conv1d(
            dimension,
            dimension,
            kernel_size=9,
            padding=4,
            groups=dimension,
            bias=False,
        )
        self.pointwise = nn.Conv1d(
            dimension,
            dimension,
            kernel_size=1,
            bias=False,
        )
        self.feedforward = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(
                dimension,
                dimension * 2,
            ),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(
                dimension * 2,
                dimension,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(x)
        attention, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        x = x + attention

        conv = self.conv_norm(x)
        conv = conv.transpose(1, 2)
        conv = self.depthwise(conv)
        conv = self.pointwise(conv)
        conv = torch.nn.functional.gelu(
            conv
        )
        x = x + conv.transpose(1, 2)

        return x + self.feedforward(x)


class EEGConformer(nn.Module):
    def __init__(
        self,
        channels: int,
        classes: int,
        *,
        dimension: int = 96,
    ) -> None:
        super().__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv1d(
                channels,
                dimension,
                kernel_size=17,
                stride=4,
                padding=8,
                bias=False,
            ),
            nn.BatchNorm1d(dimension),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(dimension)
                for _ in range(3)
            ]
        )
        self.norm = nn.LayerNorm(
            dimension
        )
        self.classifier = nn.Linear(
            dimension,
            classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.patch_embedding(
            x
        ).transpose(1, 2)
        x = x + _sinusoidal_position(
            x.shape[1],
            x.shape[2],
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0)

        for block in self.blocks:
            x = block(x)

        x = self.norm(
            x.mean(dim=1)
        )
        return self.classifier(x)


def build_model(
    architecture: str,
    *,
    channels: int,
    classes: int,
) -> nn.Module:
    architecture = architecture.lower()

    if architecture == "eegnet":
        return EEGNet(
            channels=channels,
            classes=classes,
        )
    if architecture == "cnn1d":
        return CNN1D(
            channels=channels,
            classes=classes,
        )
    if architecture == "cnn_lstm":
        return CNNLSTM(
            channels=channels,
            classes=classes,
        )
    if architecture == "temporal_cnn":
        return TemporalCNN(
            channels=channels,
            classes=classes,
        )
    if architecture == "transformer":
        return EEGTransformer(
            channels=channels,
            classes=classes,
        )
    if architecture == "eeg_conformer":
        return EEGConformer(
            channels=channels,
            classes=classes,
        )

    raise ValueError(
        "Unknown architecture "
        f"{architecture!r}. "
        f"Choose one of {ARCHITECTURES}."
    )
