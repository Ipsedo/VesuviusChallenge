# -*- coding: utf-8 -*-
from math import log
from typing import Tuple

import torch as th
from torch import nn


class Positional2dEncoding(nn.Module):
    def __init__(self, channels: int, kernel_size: Tuple[int, int]):
        super().__init__()
        assert channels % 2 == 0

        channels = channels // 2

        div_term = th.exp(
            th.arange(0, channels, 2) * th.tensor(-log(10000.0) / channels)
        )

        kernel_size_x = kernel_size[0]
        kernel_size_y = kernel_size[1]

        position_x = th.arange(kernel_size_x).unsqueeze(1)
        position_y = th.arange(kernel_size_y).unsqueeze(1)

        pe = th.zeros(channels * 2, kernel_size_x, kernel_size_y)

        pe[0:channels:2, :, :] = (
            th.sin(position_x * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, kernel_size_y, 1)
        )
        pe[1:channels:2, :, :] = (
            th.cos(position_x * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, kernel_size_y, 1)
        )

        pe[channels::2, :, :] = (
            th.sin(position_y * div_term)
            .transpose(0, 1)
            .unsqueeze(2)
            .repeat(1, 1, kernel_size_x)
        )
        pe[channels + 1 :: 2, :, :] = (
            th.cos(position_y * div_term)
            .transpose(0, 1)
            .unsqueeze(2)
            .repeat(1, 1, kernel_size_x)
        )

        self.register_buffer(
            "_pe", pe.flatten(1, 2).permute(1, 0).unsqueeze(0)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 3

        out: th.Tensor = x + self._pe[:, : x.size(1), :]

        return out
