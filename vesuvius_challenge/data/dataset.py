# -*- coding: utf-8 -*-
import re
from os import listdir
from os.path import exists, isdir, isfile, join
from typing import Tuple

import torch as th
from torch.utils.data import Dataset
from tqdm import tqdm


class VesuviusDataset(Dataset):
    def __init__(self, data_folder_path: str) -> None:
        super().__init__()

        assert exists(data_folder_path)
        assert isdir(data_folder_path)

        re_img = re.compile(r"^img_(\d+).pt$")
        re_lbl = re.compile(r"^lbl_(\d+).pt$")

        self.__img_path = sorted(
            tqdm(
                [
                    join(data_folder_path, f)
                    for f in listdir(data_folder_path)
                    if re_img.match(f) and isfile(join(data_folder_path, f))
                ]
            )
        )

        self.__lbl_path = sorted(
            tqdm(
                [
                    join(data_folder_path, f)
                    for f in listdir(data_folder_path)
                    if re_lbl.match(f) and isfile(join(data_folder_path, f))
                ]
            )
        )

        assert len(self.__img_path) == len(self.__lbl_path)

    def __getitem__(self, idx: int) -> Tuple[th.Tensor, th.Tensor]:
        img = th.load(self.__img_path[idx])
        lbl = th.load(self.__lbl_path[idx])
        return img, lbl

    def __len__(self) -> int:
        return len(self.__img_path)
