# -*- coding: utf-8 -*-
import torch as th
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from .data import ToDType, VesuviusDataset
from .options import InferOptions, ModelOptions


def infer(model_options: ModelOptions, infer_options: InferOptions) -> None:
    model = model_options.new_model()
    model.load_state_dict(th.load(infer_options.model_state_dict))

    print(f"parameters : {model.count_parameters()}")

    model.eval()

    if infer_options.cuda:
        model.cuda()

    transform = Compose([ToDType(th.float)])

    dataset = VesuviusDataset(infer_options.dataset_path, transform, transform)

    data_loader = DataLoader(
        dataset,
        batch_size=infer_options.batch_size,
        num_workers=4,
    )

    loss = 0.0

    with th.no_grad():
        idx = 0
        for img, lbl in tqdm(data_loader):
            if infer_options.cuda:
                img = img.cuda()
                lbl = lbl.cuda()

            out = model(img)

            loss += binary_cross_entropy(out, lbl, reduction="mean").item()
            idx += 1

        loss /= idx

    print(f"bce : {loss}")
