# -*- coding: utf-8 -*-
from torch import nn


def init_module(module: nn.Module) -> None:
    if isinstance(
        module,
        (
            nn.Linear,
            nn.Conv2d,
            nn.ConvTranspose2d,
            nn.Conv3d,
            nn.ConvTranspose3d,
        ),
    ):
        nn.init.xavier_normal_(module.weight, gain=1e-3)
        if module.bias is not None:
            nn.init.normal_(module.bias, std=1e-3)
    elif isinstance(
        module,
        (
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.LayerNorm,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        ),
    ):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
