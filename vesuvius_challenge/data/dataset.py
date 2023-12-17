# -*- coding: utf-8 -*-
import re
from os import listdir
from os.path import exists, isdir, isfile, join
from typing import Tuple

import torch as th
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm import tqdm


class VesuviusDataset(Dataset):
    def __init__(
        self,
        data_folder_path: str,
        img_transform: Compose,
        lbl_transform: Compose,
    ) -> None:
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

        self.__img_tr = img_transform
        self.__lbl_tr = lbl_transform

    def __getitem__(self, idx: int) -> Tuple[th.Tensor, th.Tensor]:
        img = self.__img_tr(th.load(self.__img_path[idx]))
        lbl = self.__lbl_tr(th.load(self.__lbl_path[idx]))
        img = img.squeeze(0).permute(2, 0, 1)[1:, :, :]
        lbl = th.where(lbl.squeeze(0) == 0, th.tensor(1), th.tensor(2))
        return img, lbl

    def __len__(self) -> int:
        return len(self.__img_path)
