# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List

import torch as th


class Transform(ABC):
    @abstractmethod
    def _process(self, x: th.Tensor) -> th.Tensor:
        pass

    def __call__(self, x: th.Tensor) -> th.Tensor:
        assert x.size(0) == 1

        return self._process(x)


class ToDType(Transform):
    def __init__(self, data_type: th.dtype) -> None:
        super().__init__()
        self.__data_type = data_type

    def _process(self, x: th.Tensor) -> th.Tensor:
        return x.to(self.__data_type)


class MinMaxScale(Transform):
    def __init__(self, dim: List[int], eps: float = 1e-8):
        super().__init__()
        self.__dim = dim
        self.__eps = eps

    def _process(self, x: th.Tensor) -> th.Tensor:
        x_max = th.amax(x, dim=self.__dim, keepdim=True)
        x_min = th.amin(x, dim=self.__dim, keepdim=True)

        return (x - x_min) / (x_max - x_min + self.__eps)


class RangeChange(Transform):
    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound

    def _process(self, x: th.Tensor) -> th.Tensor:
        out: th.Tensor = (
            x * (self.__upper_bound - self.__lower_bound) + self.__lower_bound
        )
        return out
