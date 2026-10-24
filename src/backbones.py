"""Common backbones for the feature-geometry/capacity study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn

from features import feature_channel_count, validate_feature_code


BackboneCode = Literal["BB1", "BB2"]
WIDTHS = (0.50, 0.75, 1.00, 1.50, 2.00)


def round_channels(value: float, multiple: int = 4, minimum: int = 8) -> int:
    if value <= 0:
        raise ValueError("Channel value must be positive")
    rounded = int(round(value / multiple) * multiple)
    return max(minimum, rounded)


def scale_channels(base: tuple[int, ...], width: float) -> tuple[int, ...]:
    if width <= 0:
        raise ValueError("width must be positive")
    return tuple(round_channels(channel * width) for channel in base)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1) -> None:
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 5, stride=stride, padding=2, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pointwise(self.depthwise(x)) + self.shortcut(x))


class StandardResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvBNAct(in_channels, out_channels, 5, stride),
            nn.Conv1d(out_channels, out_channels, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.shortcut(x))


class CommonBackbone(nn.Module):
    def __init__(self, input_channels: int, stage_channels: tuple[int, ...], block: type[nn.Module], num_classes: int = 11) -> None:
        super().__init__()
        if len(stage_channels) != 3:
            raise ValueError("The controlled backbone must have exactly three stages")
        self.stem = ConvBNAct(input_channels, stage_channels[0], 5, 1)
        self.stages = nn.Sequential(
            block(stage_channels[0], stage_channels[0], 1),
            block(stage_channels[0], stage_channels[1], 2),
            block(stage_channels[1], stage_channels[1], 1),
            block(stage_channels[1], stage_channels[2], 2),
            block(stage_channels[2], stage_channels[2], 1),
        )
        self.classifier = nn.Linear(stage_channels[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        x = x.mean(dim=-1)
        return self.classifier(x)


@dataclass(frozen=True)
class BackboneSpec:
    backbone: BackboneCode
    feature: str
    width_multiplier: float
    input_channels: int
    stage_channels: tuple[int, ...]
    num_classes: int = 11

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_model(backbone: str, feature: str, width: float, num_classes: int = 11) -> tuple[nn.Module, BackboneSpec]:
    if backbone not in {"BB1", "BB2"}:
        raise ValueError("backbone must be BB1 or BB2")
    validate_feature_code(feature)
    input_channels = feature_channel_count(feature)
    if backbone == "BB1":
        base = (24, 32, 48)
        channels = scale_channels(base, width)
        block = DepthwiseSeparableBlock
    else:
        base = (24, 32, 48)
        channels = scale_channels(base, width)
        block = StandardResidualBlock
    model = CommonBackbone(input_channels, channels, block, num_classes)
    return model, BackboneSpec(backbone, feature, float(width), input_channels, channels, num_classes)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total
