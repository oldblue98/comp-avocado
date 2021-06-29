import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from .utils import get_img

class BaseDataset(Dataset):
    def __init__(self, df, 
                transforms=None,
                image_name_col = "image_path",
                label_col="label", 
                output_label=True,
                one_hot_label=False):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.image_name_col = image_name_col
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        if output_label:
            self.labels = self.df[self.label_col].values
            if one_hot_label:
                self.labels = np.eye(self.df[self.label_col].max()+1)[self.labels]
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        if self.output_label:
            target = self.labels[index]

        img = get_img(self.df.loc[index][self.image_name_col])

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.output_label:
            return img, target
        else:
            return img