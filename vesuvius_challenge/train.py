# -*- coding: utf-8 -*-
from .options import ModelOptions


def train(model_options: ModelOptions, dataset_path: str) -> None:
    print(model_options)
    print(dataset_path)
