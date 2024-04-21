# -*- coding: utf-8 -*-

import torch as th
from torch import nn
from torch.nn import functional as F
from unfoldNd import foldNd, unfoldNd

from .positional_encoding import Positional2dEncoding


class WindowedTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: int,
        kernel_size: int,
        padding: int,
        num_heads: int = 8,
        encoder_layers: int = 3,
    ) -> None:
        super().__init__()

        self.__channels = channels
        self.__kernel_size = kernel_size
        self.__padding = padding

        self.__trf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                channels,
                num_heads,
                hidden,
                dropout=0.1,
                activation=F.mish,
                batch_first=True,
            ),
            encoder_layers,
            enable_nested_tensor=False,
        )

        self.__pe = Positional2dEncoding(channels, (kernel_size, kernel_size))

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

        input_trf = self.__pe(self.__linear_path_unfold(input_encoded))

        out: th.Tensor = (
            self.__trf(input_trf)
            .view(b, -1, self.__kernel_size ** len(sizes), self.__channels)
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
