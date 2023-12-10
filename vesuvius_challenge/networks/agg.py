# -*- coding: utf-8 -*-
from typing import List, Literal, Union

import torch as th
from torch import nn


class Agg(nn.Module):
    def __init__(
        self, fun: Literal["mean", "sum", "max"], dim: Union[int, List[int]]
    ) -> None:
        super().__init__()

        self.__dim = [dim] if isinstance(dim, int) else dim

        functions = {
            "mean": th.mean,
            "sum": th.sum,
            "max": th.amax,
        }

        self.__agg_fun = functions[fun]

    def forward(self, x: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.__agg_fun(x, dim=self.__dim)  # type: ignore
        return out
