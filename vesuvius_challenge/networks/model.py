# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple

import torch as th
from torch import nn

from .convolutions import ConvBlock, OutputConv
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

        self.__encoder = nn.Sequential(
            *[ConvBlock(c_i, c_o, num_groups, "down") for c_i, c_o in channels]
        )

        self.__target_encoder = nn.Sequential(
            *[ConvBlock(c_i, c_o, num_groups, "down") for c_i, c_o in channels]
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
                ConvBlock(c_i, c_o, num_groups, "up")
                for c_i, c_o in decoder_channels
            ]
        )

        self.__output = OutputConv(decoder_channels[-1][0], channels[0][0])

    def forward(
        self, x: th.Tensor, tgt: Optional[th.Tensor] = None
    ) -> th.Tensor:
        assert len(x.size()) == 5

        encoded_x = self.__encoder(x)

        if tgt is not None:
            assert len(tgt.size()) == len(x.size())
            assert all(x.size(i) == tgt.size(i) for i in range(len(x.size())))

            encoded_tgt = self.__target_encoder(tgt)
            out_encoded = self.__trf(encoded_x, encoded_tgt)
        else:
            out_encoded = self.__trf(encoded_x)

        out: th.Tensor = self.__decoder(out_encoded)
        out = self.__output(out)
        out = th.tanh(out.sum(dim=-1))

        return out
