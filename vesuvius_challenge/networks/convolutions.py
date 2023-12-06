# -*- coding: utf-8 -*-
from typing import Literal

from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
    ) -> None:
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                (3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                padding_mode="replicate",
            ),
            nn.Mish(),
            nn.GroupNorm(num_groups, out_channels),
        )


class StrideConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        scale: Literal["up", "down"],
    ) -> None:
        constructor = {
            "up": nn.ConvTranspose3d,
            "down": nn.Conv3d,
        }

        padding = {
            "up": "zeros",
            "down": "replicate",
        }

        super().__init__(
            constructor[scale](
                in_channels,
                out_channels,
                (4, 4, 4),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                padding_mode=padding[scale],
            ),
            nn.Mish(),
            nn.GroupNorm(num_groups, out_channels),
        )


class OutputConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                1,
                1,
                0,
            ),
        )
