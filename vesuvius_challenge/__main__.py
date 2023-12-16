# -*- coding: utf-8 -*-
import argparse
import re
from typing import List, Tuple

from .data import process_data_stride
from .infer import infer
from .options import InferOptions, ModelOptions, TrainOptions
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

    main_sub_parser = parser.add_subparsers(
        title="mode", required=True, dest="mode"
    )

    ############
    # DataPrep #
    ############

    data_parser = main_sub_parser.add_parser("data")

    data_parser.add_argument("extracted_zip_folder", type=str)
    data_parser.add_argument("output_folder", type=str)
    data_parser.add_argument("--width", type=int, default=256)
    data_parser.add_argument("--height", type=int, default=256)
    data_parser.add_argument("--images", type=int, nargs="+", default=[1, 2])
    data_parser.add_argument("--stride", type=int, nargs=2, default=[64, 64])

    ############
    # Modeling #
    ############

    model_parser = main_sub_parser.add_parser("model")

    model_parser.add_argument(
        "--channels",
        type=_channels,
        default=[(1, 4), (4, 8), (8, 16), (16, 32), (32, 64)],
    )
    model_parser.add_argument("--slices", type=int, default=65)
    model_parser.add_argument("--num-groups", type=int, default=4)
    model_parser.add_argument("--trf-kernel-size", type=int, default=3)
    model_parser.add_argument("--trf-padding", type=int, default=1)
    model_parser.add_argument("--trf-layers", type=int, default=3)
    model_parser.add_argument("--trf-hidden", type=int, default=128)
    model_parser.add_argument("--trf-num-heads", type=int, default=4)

    model_mode_parser = model_parser.add_subparsers(
        title="model_mode", required=True, dest="model_mode"
    )

    # Train

    train_parser = model_mode_parser.add_parser("train")

    train_parser.add_argument("dataset_path", type=str)
    train_parser.add_argument("output_path", type=str)
    train_parser.add_argument("--nb-epoch", type=int, default=100)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--save-every", type=int, default=1024)
    train_parser.add_argument("--metric-length", type=int, default=64)
    train_parser.add_argument("--cuda", action="store_true")

    # Infer

    infer_parser = model_mode_parser.add_parser("infer")
    infer_parser.add_argument("model_state_dict", type=str)
    infer_parser.add_argument("dataset_path", type=str)
    infer_parser.add_argument("output_path", type=str)
    infer_parser.add_argument("--batch-size", type=int, default=16)
    infer_parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    if args.mode == "model":

        model_options = ModelOptions(
            channels=args.channels,
            slices=args.slices,
            num_groups=args.num_groups,
            trf_kernel_size=args.trf_kernel_size,
            trf_padding=args.trf_padding,
            trf_layers=args.trf_layers,
            hidden=args.trf_hidden,
            num_heads=args.trf_num_heads,
        )

        if args.model_mode == "train":

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

        elif args.model_mode == "infer":
            infer_options = InferOptions(
                model_state_dict=args.model_state_dict,
                dataset_path=args.dataset_path,
                output_path=args.output_path,
                batch_size=args.batch_size,
                cuda=args.cuda,
            )

            infer(model_options, infer_options)
        else:
            model_parser.error(f'Unrecognized model_model "{args.model_mode}"')
    elif args.mode == "data":
        process_data_stride(
            args.extracted_zip_folder,
            args.output_folder,
            (args.width, args.height),
            args.stride,
            args.images,
        )
    else:
        parser.error(f'Unrecognized mode "{args.mode}"')


if __name__ == "__main__":
    main()
