"""MobileNet-V1-style classifier for 1x64x101 log-mel spectrograms.

The default channel widths are 32, 64, 128, and 128. ``width_mult`` scales
each width with a minimum of eight channels. The stem is followed by three
depthwise-separable blocks, each made from depthwise 3x3 + BN + ReLU and
pointwise 1x1 + BN + ReLU. ReLU is inexpensive and avoids saturation, which
fits the mobile inference target. Global average pooling removes the spatial
dimensions, dropout regularizes the 128-dimensional representation, and a
linear classifier emits ten logits.

Default shape diagram (C x frequency x time)::

    input       1 x 64 x 101
    stem       32 x 64 x 101       Conv3x3, stride 1
    block 1    64 x 32 x 51        depthwise 3x3 + pointwise 1x1, stride 2
    block 2   128 x 16 x 26        depthwise 3x3 + pointwise 1x1, stride 2
    block 3   128 x 16 x 26        depthwise 3x3 + pointwise 1x1, stride 1
    pooling   128 x 1 x 1          global average pooling
    dropout   128                   dropout(p=0.2)
    output     10                   linear classifier
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class DepthwiseSeparableConv(nn.Module):
    """MobileNet-V1 depthwise and pointwise convolution block."""

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
        """Apply depthwise convolution followed by pointwise projection."""
        x = self.activation(self.dw_bn(self.depthwise(x)))
        return self.activation(self.pw_bn(self.pointwise(x)))


class MobileNetStyleCNN(nn.Module):
    """Small hand-written MobileNet-style command classifier."""

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
        if width_mult <= 0:
            raise ValueError("width_mult must be positive")

        scaled_channels = [max(8, int(channel * width_mult)) for channel in channels]
        stem_out = scaled_channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_out, kernel_size=3, stride=strides[0], padding=1, bias=False),
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
        """Return class logits for a batch of mel spectrograms."""
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(self.dropout(x))

    def count_parameters(self) -> int:
        """Return the total number of trainable model parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def estimate_macs(self, input_shape: tuple[int, int, int] = (1, 64, 101)) -> int:
        """Estimate multiply-accumulates for one input using local hooks."""
        macs = 0

        def hook(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
            nonlocal macs
            if isinstance(module, nn.Conv2d):
                output_height, output_width = output.shape[-2:]
                macs += (
                    output.shape[0]
                    * output_height
                    * output_width
                    * module.out_channels
                    * (module.in_channels // module.groups)
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                )
            elif isinstance(module, nn.Linear):
                macs += inputs[0].shape[0] * module.in_features * module.out_features

        hooks = [
            module.register_forward_hook(hook)
            for module in self.modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self(torch.zeros((1, *input_shape), device=next(self.parameters()).device))
        if was_training:
            self.train()
        for registered_hook in hooks:
            registered_hook.remove()
        return macs