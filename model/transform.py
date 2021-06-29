from albumentations import (
    Compose, HorizontalFlip
)
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return Compose([
        HorizontalFlip(p=0.5),
        ToTensorV2(p=1.)
    ], p=1.)

def get_valid_transforms():
    return Compose([
        ToTensorV2(p=1.)
    ], p=1.)