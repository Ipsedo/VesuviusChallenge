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


def read_split_slice(
    img_folder: str,
    desired_size: Tuple[int, int],
    idx: int,
) -> th.Tensor:
    return (
        F.unfold(
            _to_tensor(
                Image.open(
                    join(img_folder, "surface_volume", f"{idx:02}.tif")
                ).convert("F")
            )
            .div(2.0**16 - 1.0)
            .mul(2.0)
            .sub(1.0),
            desired_size,
            1,
            0,
            desired_size,
        )
        .view(1, desired_size[0], desired_size[1], -1)
        .permute(3, 0, 1, 2)
    )


def read_split_slices(
    img_folder: str, desired_size: Tuple[int, int]
) -> List[th.Tensor]:
    return [
        read_split_slice(img_folder, desired_size, slice_idx)
        for slice_idx in tqdm(range(65))
    ]


def read_slice(
    img_folder: str,
    idx: int,
) -> th.Tensor:
    slice_t: th.Tensor = (
        _to_tensor(
            Image.open(
                join(img_folder, "surface_volume", f"{idx:02}.tif")
            ).convert("F")
        )
        .div(2.0**16 - 1.0)
        .mul(2.0)
        .sub(1.0)
    )
    return slice_t


def read_mask(img_folder: str) -> th.Tensor:
    mask_t: th.Tensor = _to_tensor(
        Image.open(join(img_folder, "mask.png"))
    ).gt(0)
    return mask_t


def read_label(img_folder: str) -> th.Tensor:
    label_t: th.Tensor = _to_tensor(
        Image.open(join(img_folder, "inklabels.png"))
    ).gt(0)
    return label_t


def process_data(
    extracted_zip_folder: str,
    output_folder: str,
    desired_size: Tuple[int, int],
    image_index: List[int],
    filter_full_lbl_patches: bool,
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

        full_lbl_t = label_t.eq(0).flatten(-2, -1).all(dim=-1) | label_t.eq(
            1
        ).flatten(-2, -1).all(dim=-1)

        for i in tqdm(range(label_t.size(0))):
            if bool(mask_t[i]):
                if filter_full_lbl_patches and bool(full_lbl_t[i]):
                    continue

                th.save(
                    label_t[i].clone(),
                    join(output_folder, f"lbl_{idx}.pt"),
                )

                th.save(
                    th.stack([s[i] for s in slices_t], dim=-1).clone(),
                    join(output_folder, f"img_{idx}.pt"),
                )

                idx += 1


def process_data_stride(
    extracted_zip_folder: str,
    output_folder: str,
    desired_size: Tuple[int, int],
    stride: Tuple[int, int],
    image_index: List[int],
) -> None:
    if not exists(output_folder):
        mkdir(output_folder)
    else:
        assert isdir(output_folder)

    idx = 0

    for img_idx in image_index:
        img_folder = join(extracted_zip_folder, "train", str(img_idx))

        label_t = read_label(img_folder)
        mask_t = read_mask(img_folder)

        slices_l = [read_slice(img_folder, i) for i in tqdm(range(65))]

        tqdm_bar = tqdm(
            [
                (k_w, k_h)
                for k_w in range(0, label_t.size(1), stride[0])
                for k_h in range(0, label_t.size(2), stride[1])
            ]
        )
        for k_w, k_h in tqdm_bar:
            patch_mask_t = mask_t[
                :, k_w : k_w + desired_size[0], k_h : k_h + desired_size[1]
            ]

            if bool(patch_mask_t.flatten(1, 2).any(1)):
                patch_label_t = label_t[
                    :, k_w : k_w + desired_size[0], k_h : k_h + desired_size[1]
                ]
                patch_slices_t = th.stack(
                    [
                        s[
                            :,
                            k_w : k_w + desired_size[0],
                            k_h : k_h + desired_size[1],
                        ]
                        for s in slices_l
                    ],
                    dim=-1,
                )

                th.save(
                    patch_label_t.clone(),
                    join(output_folder, f"lbl_{idx}.pt"),
                )

                th.save(
                    patch_slices_t.clone(),
                    join(output_folder, f"img_{idx}.pt"),
                )

                idx += 1

            tqdm_bar.set_description(f"written : {idx}")
