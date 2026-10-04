import torch
from torch import nn


class EEGNet(nn.Module):
    def __init__(self, channels: int, classes: int, *, f1: int = 8, depth_multiplier: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        f2 = f1 * depth_multiplier
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 63), padding="same", bias=False),
            nn.BatchNorm2d(f1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(f1, f2, kernel_size=(channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(f2, f2, kernel_size=(1, 15), padding="same", groups=f2, bias=False),
            nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 8)),
        )
        self.classifier = nn.Linear(f2 * 8, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("EEGNet expects [batch, channels, samples]")
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return self.classifier(x.flatten(1))
