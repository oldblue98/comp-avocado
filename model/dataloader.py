import numpy as np
import pandas as pd
import torch

from .transform import get_train_transforms, get_valid_transforms
from .dataset import BaseDataset

def prepare_dataloader(df, trn_idx, val_idx, data_root, train_bs, valid_bs, num_workers=0, label_col="label"):
    train_df = df.loc[trn_idx, :].reset_index(drop=True)
    valid_df = df.loc[val_idx, :].reset_index(drop=True)

    train_ds = BaseDataset(train_df, transforms=get_train_transforms(), image_name_col=data_root, label_col=label_col)
    valid_ds = BaseDataset(valid_df, transforms=get_valid_transforms(), image_name_col=data_root, label_col=label_col)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_bs,
        pin_memory=True, # faster and use memory
        shuffle=True,
        num_workers=num_workers
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=valid_bs,
        pin_memory=True, # faster and use memory
        shuffle=True,
        num_workers=num_workers
    )
    return train_loader, valid_loader