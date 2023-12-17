# -*- coding: utf-8 -*-
from typing import List, NamedTuple, Tuple

from .networks import WindowedTransformer


class ModelOptions(NamedTuple):
    channels: List[Tuple[int, int]]
    slices: int
    num_groups: int
    trf_kernel_size: int
    trf_padding: int
    trf_layers: int
    hidden: int
    num_heads: int

    def new_model(self) -> WindowedTransformer:
        """return TrfAutoEncoder(
            self.channels,
            self.slices,
            self.num_groups,
            self.trf_kernel_size,
            self.trf_padding,
            self.trf_layers,
            self.hidden,
            self.num_heads,
        )"""
        return WindowedTransformer(
            self.slices,
            2,
            self.hidden,
            self.trf_kernel_size,
            self.trf_padding,
            self.num_heads,
            self.trf_layers,
            self.trf_layers,
        )


class TrainOptions(NamedTuple):
    dataset_path: str
    output_path: str
    nb_epoch: int
    learning_rate: float
    batch_size: int
    save_every: int
    metric_length: int
    cuda: bool


class InferOptions(NamedTuple):
    model_state_dict: str
    dataset_path: str
    output_path: str
    batch_size: int
    cuda: bool
