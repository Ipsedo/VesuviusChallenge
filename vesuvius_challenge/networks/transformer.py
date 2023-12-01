# -*- coding: utf-8 -*-
from typing import Optional

import torch as th
from torch import nn
from torch.nn import functional as F


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
        )

    def __linear_path_unfold(self, t: th.Tensor) -> th.Tensor:
        b = t.size(0)
        return (
            F.unfold(
                t,
                self.__kernel_size,
                dilation=1,
                padding=self.__padding,
                stride=1,
            )
            .view(b, self.__channels, self.__kernel_size**2, -1)
            # batch, patch, kernel, channels
            .permute(0, 3, 2, 1)
            .contiguous()
            .view(-1, self.__kernel_size**2, self.__channels)
        )

    def __generate(self, input_trf: th.Tensor) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        b, _, c = input_trf.size()

        # start token
        target = th.zeros((b, 1, c), device=device)

        for _ in range(self.__kernel_size**2):
            out = self.__trf(input_trf, target)
            target = th.cat([target, out[:, -1, None, :]], dim=1)

        # remove start token
        return target[:, 1:, :]

    def forward(
        self,
        input_encoded: th.Tensor,
        target_encoded: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        assert len(input_encoded.size()) == 4

        b, _, w, h = input_encoded.size()

        input_trf = self.__linear_path_unfold(input_encoded)

        if target_encoded is not None:
            # training
            assert len(target_encoded.size()) == 4
            assert all(
                input_encoded.size(i) == target_encoded.size(i)
                for i in range(len(input_encoded.size()))
            )

            device = "cuda" if next(self.parameters()).is_cuda else "cpu"

            target_trf = self.__linear_path_unfold(target_encoded)

            # start token
            target_trf = th.cat(
                [
                    th.zeros(
                        target_trf.size(0), 1, self.__channels, device=device
                    ),
                    target_trf,
                ],
                dim=1,
            )

            out: th.Tensor = self.__trf(input_trf, target_trf)

            # remove end token
            out = out[:, :-1, :]
        else:
            # auto-regressive generation
            out = self.__generate(input_trf)

        out = (
            out.view(b, -1, self.__kernel_size**2, self.__channels)
            .permute(0, 3, 2, 1)
            .contiguous()
            .view(b, self.__channels * self.__kernel_size**2, -1)
        )

        out = F.fold(
            out,
            (w, h),
            self.__kernel_size,
            dilation=1,
            padding=self.__padding,
            stride=1,
        )

        return out
