# -*- coding: utf-8 -*-

from torch import nn


class Conv3dBlock(nn.Sequential):
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
            nn.BatchNorm3d(out_channels),
        )


class DownConv3dBlock(nn.Sequential):
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
                (4, 4, 4),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                padding_mode="replicate",
            ),
            nn.Mish(),
            nn.BatchNorm3d(out_channels),
        )


class DownConv2dBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                (4, 4),
                stride=(2, 2),
                padding=(1, 1),
                padding_mode="replicate",
            ),
            nn.Mish(),
            nn.BatchNorm2d(out_channels),
        )


class Conv2dBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                (3, 3),
                (1, 1),
                (1, 1),
                padding_mode="replicate",
            ),
            nn.Mish(),
            nn.BatchNorm2d(out_channels),
        )


class UpConv2dBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
    ) -> None:
        super().__init__(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                (4, 4),
                (2, 2),
                (1, 1),
                padding_mode="zeros",
            ),
            nn.Mish(),
            nn.BatchNorm2d(out_channels),
        )


class OutputConv2d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                1,
                1,
                0,
            ),
        )
