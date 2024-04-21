# -*- coding: utf-8 -*-
from statistics import mean
from typing import List, Tuple

import numpy as np
import torch as th
from torch import nn

from .agg import Agg
from .convolutions import (
    Conv2dBlock,
    Conv3dBlock,
    DownConv3dBlock,
    OutputConv2d,
    UpConv2dBlock,
)
from .transformer import WindowedTransformer


class TrfAutoEncoder(nn.Module):
    def __init__(
        self,
        channels: List[Tuple[int, int]],
        slices: int,
        num_groups: int,
        trf_kernel_size: int,
        trf_padding: int,
        trf_layers: int,
        hidden: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.__slices = slices

        self.__input_encoder = nn.ModuleList(
            nn.Sequential(
                Conv3dBlock(c_i, c_o, num_groups),
                DownConv3dBlock(c_o, c_o, num_groups),
            )
            for c_i, c_o in channels
        )

        self.__flat = Agg("mean", dim=-1)

        self.__trf = WindowedTransformer(
            channels[-1][1],
            hidden,
            trf_kernel_size,
            trf_padding,
            num_heads,
            trf_layers,
        )

        decoder_channels = [(c_o, c_i) for c_i, c_o in reversed(channels)]
        decoder_channels[-1] = (
            decoder_channels[-1][0],
            decoder_channels[-1][0],
        )

        self.__to_decoder = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.__slices // 2 ** (i + 1), 1),
                nn.Flatten(-2, -1),
            )
            for i in reversed(range(len(channels)))
        )

        self.__decoder = nn.ModuleList(
            nn.Sequential(
                Conv2dBlock(c_i, c_o, num_groups),
                UpConv2dBlock(c_o, c_o, num_groups),
            )
            for c_i, c_o in decoder_channels
        )

        c_i = decoder_channels[-1][0]
        c_o = channels[0][0]
        self.__output = nn.Sequential(
            Conv2dBlock(c_i, c_i, num_groups),
            OutputConv2d(c_i, c_o),
            nn.Sigmoid(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 5

        out = x
        bypasses = []
        for enc in self.__input_encoder:
            out = enc(out)
            bypasses.append(out)

        out = self.__flat(out)

        out = self.__trf(out)

        for dec, bypass, to_dec in zip(
            self.__decoder, reversed(bypasses), self.__to_decoder
        ):
            bypass = to_dec(bypass)
            out = out + bypass
            out = dec(out)

        out = self.__output(out)

        return out

    def count_parameters(self) -> int:
        return int(
            sum(
                np.prod(p.size()) for p in self.parameters() if p.requires_grad
            )
        )

    def grad_norm(self) -> float:
        return float(
            mean(
                p.grad.norm().item()
                for p in self.parameters()
                if p.grad is not None
            )
        )
