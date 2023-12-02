# -*- coding: utf-8 -*-
from os import mkdir
from os.path import exists, isdir, join
from typing import List, Tuple

import torch as th
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import ToTensor
from tqdm import tqdm

_to_tensor = ToTensor()


def read_split_label(
    img_folder: str, desired_size: Tuple[int, int]
) -> th.Tensor:
    return (
        F.unfold(
            _to_tensor(Image.open(join(img_folder, "inklabels.png")))[None],
            desired_size,
            1,
            0,
            desired_size,
        )
        .view(1, desired_size[0], desired_size[1], -1)
        .permute(3, 0, 1, 2)
        .gt(0)
    )


def read_split_mask(
    img_folder: str, desired_size: Tuple[int, int]
) -> th.Tensor:
    return (
        F.unfold(
            _to_tensor(Image.open(join(img_folder, "mask.png")))[None],
            desired_size,
            1,
            0,
            desired_size,
        )
        .view(desired_size[0] * desired_size[1], -1)
        .permute(1, 0)
        .gt(0)
        .any(dim=1)
    )


def read_split_slices(
    img_folder: str, desired_size: Tuple[int, int]
) -> List[th.Tensor]:
    return [
        F.unfold(
            _to_tensor(
                Image.open(
                    join(img_folder, "surface_volume", f"{slice_idx:02}.tif")
                )
            ).to(th.float32),
            desired_size,
            1,
            0,
            desired_size,
        )
        .to(th.int16)
        .view(1, desired_size[0], desired_size[1], -1)
        .permute(3, 0, 1, 2)
        for slice_idx in tqdm(range(1, 65))
    ]


def process_data(
    extracted_zip_folder: str,
    output_folder: str,
    desired_size: Tuple[int, int],
    image_index: List[int],
) -> None:
    if not exists(output_folder):
        mkdir(output_folder)
    else:
        assert isdir(output_folder)

    idx = 0

    for img_idx in image_index:
        img_folder = join(extracted_zip_folder, "train", str(img_idx))

        label_t = read_split_label(img_folder, desired_size)
        mask_t = read_split_mask(img_folder, desired_size)
        slices_t = read_split_slices(img_folder, desired_size)

        assert label_t.size(0) == mask_t.size(0)
        assert all(mask_t.size(0) == s.size(0) for s in slices_t)

        for i in tqdm(range(label_t.size(0))):
            if bool(mask_t[i]):
                th.save(
                    label_t[i].clone(),
                    join(output_folder, f"lbl_{idx}.pt"),
                )

                th.save(
                    th.stack([s[i] for s in slices_t], dim=-1).clone(),
                    join(output_folder, f"img_{img_idx}.pt"),
                )

                idx += 1
