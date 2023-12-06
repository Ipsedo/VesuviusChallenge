# -*- coding: utf-8 -*-
from statistics import mean
from typing import List, Optional, Tuple

import numpy as np
import torch as th
from torch import nn

from .convolutions import ConvBlock, OutputConv, StrideConvBlock
from .transformer import WindowedTransformer


class TrfAutoEncoder(nn.Module):
    def __init__(
        self,
        channels: List[Tuple[int, int]],
        num_groups: int,
        trf_kernel_size: int,
        trf_padding: int,
        trf_layers: int,
        hidden: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.__input = ConvBlock(1, channels[0][0], num_groups)
        self.__target = ConvBlock(1, channels[0][0], num_groups)

        self.__encoder = nn.Sequential(
            *[
                nn.Sequential(
                    ConvBlock(c_i, c_o, num_groups),
                    StrideConvBlock(c_o, c_o, num_groups, "down"),
                    # nn.MaxPool3d(2, 2),
                )
                for c_i, c_o in channels
            ]
        )

        self.__trf = WindowedTransformer(
            channels[-1][1],
            hidden,
            trf_kernel_size,
            trf_padding,
            num_heads,
            trf_layers,
            trf_layers,
        )

        decoder_channels = [(c_o, c_i) for c_i, c_o in reversed(channels)]
        decoder_channels[-1] = (
            decoder_channels[-1][0],
            decoder_channels[-1][0],
        )

        self.__decoder = nn.Sequential(
            *[
                nn.Sequential(
                    ConvBlock(c_i, c_o, num_groups),
                    StrideConvBlock(c_o, c_o, num_groups, "up"),
                    # nn.Upsample(scale_factor=2.),
                )
                for c_i, c_o in decoder_channels
            ]
        )

        c_o = decoder_channels[-1][0]
        self.__output = nn.Sequential(
            ConvBlock(c_o, c_o, num_groups),
            OutputConv(c_o, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: th.Tensor, tgt: Optional[th.Tensor] = None
    ) -> th.Tensor:
        assert len(x.size()) == 5

        encoded_x = self.__input(x)
        encoded_x = self.__encoder(encoded_x)

        if tgt is not None:
            assert len(tgt.size()) == len(x.size())
            assert all(x.size(i) == tgt.size(i) for i in range(len(x.size())))

            encoded_tgt = self.__target(tgt)
            encoded_tgt = self.__encoder(encoded_tgt)
            out_encoded = self.__trf(encoded_x, encoded_tgt)
        else:
            out_encoded = self.__trf(encoded_x)

        out: th.Tensor = self.__decoder(out_encoded)
        out = self.__output(out).log().sum(dim=-1).exp()

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
