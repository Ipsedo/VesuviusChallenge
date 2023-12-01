# -*- coding: utf-8 -*-
from typing import Tuple

import pytest
import torch as th

from vesuvius_challenge.networks.transformer import WindowedTransformer


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("channels", [4, 8])
@pytest.mark.parametrize("hidden", [4, 6])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("num_heads", [2, 4])
@pytest.mark.parametrize("sizes", [(16, 16), (16, 32)])
def test_windowed_transformer(
    batch_size: int,
    channels: int,
    hidden: int,
    kernel_size: int,
    num_heads: int,
    sizes: Tuple[int, int],
) -> None:
    trf = WindowedTransformer(
        channels, hidden, kernel_size, kernel_size // 2, num_heads, 3, 3
    )

    x = th.rand(batch_size, channels, *sizes)
    y = th.rand(batch_size, channels, *sizes)

    out = trf(x, y)

    assert len(out.size()) == 4
    assert out.size(0) == batch_size
    assert out.size(1) == channels
    assert out.size(2) == sizes[0]
    assert out.size(3) == sizes[1]

    out_gen = trf(x)

    assert len(out_gen.size()) == 4
    assert out_gen.size(0) == batch_size
    assert out_gen.size(1) == channels
    assert out_gen.size(2) == sizes[0]
    assert out_gen.size(3) == sizes[1]
