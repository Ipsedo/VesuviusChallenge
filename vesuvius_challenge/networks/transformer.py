# -*- coding: utf-8 -*-
from math import log

import torch as th
from torch import nn
from unfoldNd import foldNd, unfoldNd


class WindowedTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        nb_dim: int,
        hidden: int,
        kernel_size: int,
        padding: int,
        num_heads: int = 8,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
    ) -> None:
        super().__init__()

        self.__channels = channels
        self.__nb_dim = nb_dim
        self.__kernel_size = kernel_size
        self.__padding = padding

        self.__trf = nn.Transformer(
            channels,
            nhead=num_heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=hidden,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )

        position = th.arange(kernel_size**self.__nb_dim).unsqueeze(1)
        div_term = th.exp(
            th.arange(0, channels, 2) * th.tensor(-log(10000.0) / channels)
        )
        pe = th.zeros(1, kernel_size**self.__nb_dim, channels)
        pe[0, :, 0::2] = th.sin(position * div_term)
        pe[0, :, 1::2] = th.cos(position * div_term)
        self.register_buffer("_pe", pe)

        self.__start_pixel = nn.Parameter(th.randn((1, 1, channels)))

    def __linear_path_unfold(self, t: th.Tensor) -> th.Tensor:
        b = t.size(0)
        sizes = t.size()[2:]

        out: th.Tensor = (
            unfoldNd(
                t,
                self.__kernel_size,
                dilation=1,
                padding=self.__padding,
                stride=1,
            )
            .view(b, self.__channels, self.__kernel_size ** len(sizes), -1)
            # batch, patch, kernel, channels
            .permute(0, 3, 2, 1)
            .contiguous()
            .view(-1, self.__kernel_size ** len(sizes), self.__channels)
        )

        return out

    def forward(
        self,
        input_encoded: th.Tensor,
    ) -> th.Tensor:
        assert len(input_encoded.size()) >= 3

        b = input_encoded.size(0)
        sizes = input_encoded.size()[2:]

        input_trf = self.__linear_path_unfold(input_encoded) + self._pe

        tgt = self.__start_pixel.repeat(input_trf.size(0), 1, 1)

        for _ in range(self.__kernel_size**self.__nb_dim):
            tgt_pred = self.__trf(
                input_trf, tgt + self._pe[:, : tgt.size(1), :]
            )
            tgt = th.cat([tgt, tgt_pred[:, -1, None, :]], dim=1)

        tgt = tgt[:, 1:, :]

        out: th.Tensor = (
            tgt.view(b, -1, self.__kernel_size ** len(sizes), self.__channels)
            # batch, channels, kernel, patchs
            .permute(0, 3, 2, 1)
            .contiguous()
            .view(b, self.__channels * self.__kernel_size ** len(sizes), -1)
        )

        out = foldNd(
            out,
            sizes,
            self.__kernel_size,
            dilation=1,
            padding=self.__padding,
            stride=1,
        )

        return out
