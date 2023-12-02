# -*- coding: utf-8 -*-
import argparse
import re
from typing import List, Tuple

from .options import ModelOptions, TrainOptions
from .train import train


def _channels(string: str) -> List[Tuple[int, int]]:
    regex_match = re.compile(
        r"^ *\[(?: *\( *\d+ *, *\d+ *\) *,)* *\( *\d+ *, *\d+ *\) *] *$"
    )
    regex_layer = re.compile(r"\( *\d+ *, *\d+ *\)")
    regex_channel = re.compile(r"\d+")

    assert regex_match.match(string), "usage : [(10, 20), (20, 40), ...]"

    def _match_channels(layer_str: str) -> Tuple[int, int]:
        matched = regex_channel.findall(layer_str)
        assert len(matched) == 2
        return int(matched[0]), int(matched[1])

    return [_match_channels(layer) for layer in regex_layer.findall(string)]


def main() -> None:
    parser = argparse.ArgumentParser("vesuvius_challenge main")

    parser.add_argument(
        "--channels",
        type=_channels,
        default=[(1, 8), (8, 16), (16, 32), (32, 64)],
    )
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--trf-kernel-size", type=int, default=3)
    parser.add_argument("--trf-padding", type=int, default=1)
    parser.add_argument("--trf-layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=80)
    parser.add_argument("--num-heads", type=int, default=4)

    sub_parser = parser.add_subparsers(title="mode", required=True)

    train_parser = sub_parser.add_parser("train")
    train_parser.add_argument("dataset_path", type=str)
    train_parser.add_argument("output_path", type=str)
    train_parser.add_argument("--nb-epoch", type=int, default=100)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--save-every", type=int, default=1024)
    train_parser.add_argument("--metric-length", type=int, default=64)
    train_parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    model_options = ModelOptions(
        channels=args.channels,
        num_groups=args.num_groups,
        trf_kernel_size=args.trf_kernel_size,
        trf_padding=args.trf_padding,
        trf_layers=args.trf_layers,
        hidden=args.hidden,
        num_heads=args.num_heads,
    )

    train_options = TrainOptions(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        nb_epoch=args.nb_epoch,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        save_every=args.save_every,
        metric_length=args.metric_length,
        cuda=args.cuda,
    )

    train(model_options, train_options)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
