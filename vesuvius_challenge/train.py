# -*- coding: utf-8 -*-
from os import mkdir
from os.path import exists, isdir, join
from statistics import mean

import torch as th
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from .data import ToDType, VesuviusDataset
from .options import ModelOptions, TrainOptions


def train(model_options: ModelOptions, train_options: TrainOptions) -> None:

    if not exists(train_options.output_path):
        mkdir(train_options.output_path)
    else:
        assert isdir(train_options.output_path)

    model = model_options.new_model()

    print(f"parameters : {model.count_parameters()}")

    img_transform = Compose(
        [
            ToDType(th.float),
            # data is already [-1; 1]
        ]
    )

    lbl_transform = Compose(
        [
            ToDType(th.float),
            # data is already {0, 1}
        ]
    )

    data_loader = DataLoader(
        VesuviusDataset(
            train_options.dataset_path, img_transform, lbl_transform
        ),
        batch_size=train_options.batch_size,
        shuffle=True,
        num_workers=6,
    )

    optim = Adam(
        model.parameters(),
        lr=train_options.learning_rate,
    )

    if train_options.cuda:
        model.cuda()

    iter_idx = 0
    save_idx = 0
    bce_metrics = [1.0] * train_options.metric_length

    for e in range(train_options.nb_epoch):

        tqdm_bar = tqdm(data_loader)

        for x, y in tqdm_bar:
            if train_options.cuda:
                x = x.cuda()
                y = y.cuda()

            tgt = y.unsqueeze(-1).repeat(1, 1, 1, 1, x.size(-1))
            out = model(x, tgt)

            loss = binary_cross_entropy(out, y, reduction="mean")
            loss = loss.mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            del bce_metrics[0]
            bce_metrics.append(loss.item())

            tqdm_bar.set_description(
                f"Epoch {e:03} "
                f"- save {save_idx - 1:03} "
                f"[{iter_idx % train_options.save_every} "
                f"/ {train_options.save_every}]: "
                f"bce = {mean(bce_metrics):.5f}, "
                f"grad_norm = {model.grad_norm():.5f}"
            )

            iter_idx += 1

            if iter_idx % train_options.save_every == 0:

                th.save(
                    model.state_dict(),
                    join(train_options.output_path, f"model_{save_idx}.pt"),
                )

                th.save(
                    optim.state_dict,
                    join(train_options.output_path, f"optim_{save_idx}.pt"),
                )

                save_idx += 1
