# -*- coding: utf-8 -*-
from typing import Literal

from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            weight_norm(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    (3, 3, 3),
                    stride=(1, 1, 1),
                    padding=(1, 1, 1),
                )
            ),
            nn.Mish(),
        )


class StrideConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: Literal["up", "down"],
    ) -> None:
        constructor = {
            "up": nn.ConvTranspose3d,
            "down": nn.Conv3d,
        }
        super().__init__(
            weight_norm(
                constructor[scale](
                    in_channels,
                    out_channels,
                    (4, 4, 4),
                    stride=(2, 2, 2),
                    padding=(1, 1, 1),
                )
            ),
            nn.Mish(),
        )


class OutputConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            weight_norm(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    1,
                    1,
                    0,
                )
            ),
        )
