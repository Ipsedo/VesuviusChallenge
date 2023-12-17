# -*- coding: utf-8 -*-
from math import log
from statistics import mean
from typing import Optional, Tuple

import numpy as np
import torch as th
from torch import nn
from unfoldNd import foldNd, unfoldNd


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

        self.__pe = Positional2dEncoding(channels, (kernel_size, kernel_size))

        self.__emb = nn.Embedding(3, channels)
        self.__to_emb = nn.Sequential(
            nn.Linear(channels, 3),
            nn.Softmax(dim=-1),
        )

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
        x: th.Tensor,
        tgt: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        assert len(x.size()) >= 3

        b = x.size(0)
        sizes = x.size()[2:]

        input_trf = self.__pe(self.__linear_path_unfold(x))

        start_token = self.__emb(
            th.zeros(
                (input_trf.size(0), 1), dtype=th.long, device=input_trf.device
            )
        )

        if tgt is None:
            tgt = start_token
            for _ in range(self.__kernel_size**self.__nb_dim):
                tgt_next = self.__trf(input_trf, self.__pe(tgt))
                tgt_next = self.__to_emb(tgt_next)
                tgt_next = th.argmax(tgt_next, dim=-1)
                tgt_next = self.__emb(tgt_next)
                tgt = th.cat([tgt, tgt_next[:, -1, None, :]], dim=1)

            out = tgt[:, 1:, :]
        else:
            tgt = self.__emb(tgt).permute(0, 3, 1, 2)
            tgt = self.__linear_path_unfold(tgt)
            tgt = th.cat([start_token, tgt[:, 1:, :]], dim=1)

            tgt_mask = self.__trf.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
            out = self.__trf(input_trf, self.__pe(tgt), tgt_mask=tgt_mask)

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
