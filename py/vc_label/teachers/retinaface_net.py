"""RetinaFace (ResNet-50) network, ported from biubug6/Pytorch_Retinaface (MIT License).

Only the layers needed to load `Resnet50_Final.pth` strictly and run inference are kept.
Copyright (c) 2019 biubug6. Licensed under the MIT License.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter

_ANCHORS_PER_CELL = 2
_CHANNELS = 256


def _conv_bn(inp: int, oup: int, leaky: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def _conv_bn_no_relu(inp: int, oup: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inp, oup, 3, 1, 1, bias=False), nn.BatchNorm2d(oup))


def _conv_bn_1x1(inp: int, oup: int, leaky: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class _SSH(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3X3 = _conv_bn_no_relu(channels, channels // 2)
        self.conv5X5_1 = _conv_bn(channels, channels // 4)
        self.conv5X5_2 = _conv_bn_no_relu(channels // 4, channels // 4)
        self.conv7X7_2 = _conv_bn(channels // 4, channels // 4)
        self.conv7x7_3 = _conv_bn_no_relu(channels // 4, channels // 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv5x5_1 = self.conv5X5_1(x)
        out = torch.cat(
            [
                self.conv3X3(x),
                self.conv5X5_2(conv5x5_1),
                self.conv7x7_3(self.conv7X7_2(conv5x5_1)),
            ],
            dim=1,
        )
        return F.relu(out)


class _FPN(nn.Module):
    def __init__(self, in_channels: list[int], channels: int) -> None:
        super().__init__()
        self.output1 = _conv_bn_1x1(in_channels[0], channels)
        self.output2 = _conv_bn_1x1(in_channels[1], channels)
        self.output3 = _conv_bn_1x1(in_channels[2], channels)
        self.merge1 = _conv_bn(channels, channels)
        self.merge2 = _conv_bn(channels, channels)

    def forward(self, features: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        c3, c4, c5 = list(features.values())
        out3 = self.output3(c5)
        out2 = self.output2(c4) + F.interpolate(out3, size=c4.shape[2:], mode="nearest")
        out2 = self.merge2(out2)
        out1 = self.output1(c3) + F.interpolate(out2, size=c3.shape[2:], mode="nearest")
        out1 = self.merge1(out1)
        return [out1, out2, out3]


class _Head(nn.Module):
    def __init__(self, values_per_anchor: int) -> None:
        super().__init__()
        self.values = values_per_anchor
        self.conv1x1 = nn.Conv2d(_CHANNELS, _ANCHORS_PER_CELL * values_per_anchor, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, self.values)


class RetinaFaceNet(nn.Module):
    """Returns (box deltas [N, A, 4], face probability [N, A]) for A priors."""

    def __init__(self) -> None:
        super().__init__()
        self.body = IntermediateLayerGetter(
            resnet50(weights=None), {"layer2": 1, "layer3": 2, "layer4": 3}
        )
        self.fpn = _FPN([512, 1024, 2048], _CHANNELS)
        self.ssh1 = _SSH(_CHANNELS)
        self.ssh2 = _SSH(_CHANNELS)
        self.ssh3 = _SSH(_CHANNELS)
        self.ClassHead = nn.ModuleList(_Head(2) for _ in range(3))
        self.BboxHead = nn.ModuleList(_Head(4) for _ in range(3))
        # Unused at inference, kept so the released checkpoint loads strictly.
        self.LandmarkHead = nn.ModuleList(_Head(10) for _ in range(3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pyramid = self.fpn(self.body(x))
        features = [self.ssh1(pyramid[0]), self.ssh2(pyramid[1]), self.ssh3(pyramid[2])]
        deltas = torch.cat([head(f) for head, f in zip(self.BboxHead, features)], dim=1)
        logits = torch.cat([head(f) for head, f in zip(self.ClassHead, features)], dim=1)
        return deltas, F.softmax(logits, dim=-1)[..., 1]


def load_retinaface(weights_path: str, device: torch.device) -> RetinaFaceNet:
    """Loads Resnet50_Final.pth (keys prefixed with 'module.') with weights_only=True."""
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    net = RetinaFaceNet()
    net.load_state_dict(state, strict=True)
    return net.eval().to(device)
