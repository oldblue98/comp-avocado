import os

import pandas as pd
import numpy as np
import cv2

def get_img(path):
    '''
    pathからimgを取得する
    '''
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(path)
    img_rgb = img_bgr[..., ::-1]
    return img_rgb

def load_table(path, load_columns, target_name=None):
    df = pd.DataFrame()
    for column_name in load_columns:
        tmp_df = pd.read_feather(os.path.join(path, column_name+".feather"))
        df = pd.concat([df, tmp_df], axis=1)
    
    if target_name:
        label = pd.read_feather(os.path.join(path, target_name+".feather"))
        return df, label
    else:
        return df