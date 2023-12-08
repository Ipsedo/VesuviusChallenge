# -*- coding: utf-8 -*-
from math import log
from typing import Optional

import torch as th
from torch import nn
from unfoldNd import foldNd, unfoldNd


class WindowedTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: int,
        kernel_size: int,
        padding: int,
        num_heads: int = 8,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
    ) -> None:
        super().__init__()

        self.__channels = channels
        self.__kernel_size = kernel_size
        self.__padding = padding

        self.__trf = nn.Transformer(
            channels,
            nhead=num_heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=hidden,
            batch_first=True,
            activation="relu",
        )

        position = th.arange(kernel_size**3).unsqueeze(1)
        div_term = th.exp(
            th.arange(0, channels, 2) * (-log(10000.0) / channels)
        )
        pe = th.zeros(1, kernel_size**3, channels)
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

    def __generate(self, input_trf: th.Tensor, nb_dim: int) -> th.Tensor:
        input_trf = input_trf + self._pe

        # start token
        target = self.__start_pixel.repeat(input_trf.size(0), 1, 1)

        for _ in range(self.__kernel_size**nb_dim):
            out = self.__trf(
                input_trf, target + self._pe[:, : target.size(1), :]
            )
            target = th.cat([target, out[:, -1, None, :]], dim=1)

        # remove start token
        return target[:, 1:, :]

    def forward(
        self,
        input_encoded: th.Tensor,
        target_encoded: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        assert len(input_encoded.size()) >= 3

        b = input_encoded.size(0)
        sizes = input_encoded.size()[2:]

        input_trf = self.__linear_path_unfold(input_encoded) + self._pe

        if target_encoded is not None:
            # training
            assert len(target_encoded.size()) == len(input_encoded.size())
            assert all(
                input_encoded.size(i) == target_encoded.size(i)
                for i in range(len(input_encoded.size()))
            )

            target_trf = th.cat(
                [
                    self.__start_pixel.repeat(input_trf.size(0), 1, 1),
                    self.__linear_path_unfold(target_encoded)[:, :-1, :],
                ],
                dim=1,
            )

            target_trf = target_trf + self._pe

            tgt_mask = self.__trf.generate_square_subsequent_mask(
                target_trf.size(1), device=input_trf.device
            )

            out: th.Tensor = self.__trf(
                input_trf, target_trf, tgt_mask=tgt_mask
            )
        else:
            # auto-regressive generation
            out = self.__generate(input_trf, len(sizes))

        out = (
            out.view(b, -1, self.__kernel_size ** len(sizes), self.__channels)
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
