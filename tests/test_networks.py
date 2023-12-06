# -*- coding: utf-8 -*-
from typing import List

import pytest
import torch as th

from vesuvius_challenge.networks.model import TrfAutoEncoder
from vesuvius_challenge.networks.transformer import WindowedTransformer


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [2, 4])
@pytest.mark.parametrize("hidden", [2, 4])
@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("sizes", [[8, 8], [8, 8, 8]])
def test_windowed_transformer(
    batch_size: int,
    channels: int,
    hidden: int,
    kernel_size: int,
    num_heads: int,
    sizes: List[int],
) -> None:
    trf = WindowedTransformer(
        channels, hidden, kernel_size, kernel_size // 2, num_heads, 3, 3
    )

    x = th.rand(batch_size, channels, *sizes)
    y = th.rand(batch_size, channels, *sizes)

    out = trf(x, y)

    assert len(out.size()) == len(x.size())
    assert out.size(0) == batch_size
    assert out.size(1) == channels
    assert all(s == s_expected for s, s_expected in zip(out.size()[2:], sizes))

    out_gen = trf(x)

    assert len(out_gen.size()) == len(x.size())
    assert out_gen.size(0) == batch_size
    assert out_gen.size(1) == channels
    assert all(
        s == s_expected for s, s_expected in zip(out_gen.size()[2:], sizes)
    )


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("sizes", [[16, 16, 16], [8, 8, 8]])
def test_model(batch_size: int, sizes: List[int]) -> None:
    trf_ae = TrfAutoEncoder(
        sizes[-1],
        [(2, 4), (4, 8)],
        2,
        3,
        1,
        2,
        8,
        2,
    )

    x = th.rand(batch_size, 1, *sizes)
    tgt = th.rand(batch_size, 1, *sizes)

    out = trf_ae(x, tgt)

    assert len(out.size()) == 4
    assert out.size(0) == batch_size
    assert out.size(1) == 1
    assert out.size(2) == sizes[0]
    assert out.size(3) == sizes[1]

    out_gen = trf_ae(x)

    assert len(out_gen.size()) == 4
    assert out_gen.size(0) == batch_size
    assert out_gen.size(1) == 1
    assert out_gen.size(2) == sizes[0]
    assert out_gen.size(3) == sizes[1]
