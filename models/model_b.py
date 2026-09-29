from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.dw_bn(self.depthwise(x)))
        x = self.activation(self.pw_bn(self.pointwise(x)))
        return x


class MobileNetStyleCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        channels: Sequence[int] = (32, 64, 128, 128),
        strides: Sequence[int] = (1, 2, 2, 1),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if len(channels) != len(strides):
            raise ValueError("channels and strides must have the same length")

        scaled_channels = [max(8, int(c * width_mult)) for c in channels]

        stem_out = scaled_channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        in_channels = stem_out
        for out_channels, stride in zip(scaled_channels[1:], strides[1:]):
            blocks.append(DepthwiseSeparableConv(in_channels, out_channels, stride=stride))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)
