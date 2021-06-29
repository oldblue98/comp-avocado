import argparse
import json
import os
import datetime
import numpy as np
import pandas as pd
import torch
import torch as nn

from model.utils import load_table
from model.model import train_and_predict
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
parser.add_argument('--device', default='0')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--f', action='store_true')
options = parser.parse_args()
CFG = json.load(open(options.config))
device = torch.device(f'cuda:{options.device}')

## loggerの設定
from logging import debug, getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO
logger =getLogger("logger")
logger.setLevel(DEBUG)
## StreamHandlerの設定
handler1 = StreamHandler()
handler1.setLevel(DEBUG)
handler1.setFormatter(Formatter("%(asctime)s: %(message)s"))
## FileHandlerの設定
config_filename = os.path.splitext(os.path.basename(options.config))[0]
handler2 = FileHandler(filename=f'./logs/{config_filename}.log')
handler2.setLevel(DEBUG)
handler2.setFormatter(Formatter("%(asctime)s: %(message)s"))
#loggerに2つのハンドラを設定
logger.addHandler(handler1)
logger.addHandler(handler2)

os.makedirs(f'./data/output/{config_filename}', exist_ok=True)

def main():
    # configのログ
    logger.debug(CFG)

    # データのロード
    train, y_train_all = load_table(path="./features/feather/train", load_columns=CFG["columns"], target_name=CFG["target_name"])
    test = load_table(path="./features/feather/test", load_columns=CFG["columns"])
    qcut_target = pd.qcut(y_train_all.iloc[:, 0], CFG['q_splits'], labels=False)

    if CFG["fold_name"] == "SG":
        folds = StratifiedGroupKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), qcut_target, train.year)
    elif CFG["fold_name"] == "S":
        folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), qcut_target)
    elif CFG["fold_name"] == "K":
        folds = KFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), y_train_all)
    else:
        print("Fold is not defined")
        return

    valid_indexes = []
    preds_valid = pd.DataFrame()
    preds_test = pd.DataFrame()

    # yearの削除
    train.drop("year", axis=1, inplace=True)
    test.drop("year", axis=1, inplace=True)

    # if not options.f:
    #     for model_name in CFG["base_models"]:
    #         if os.path.isfile(f'./data/output/{config_filename}/{model_name}_train.feather'):
    #             CFG["base_models"]["_"+model_name] = CFG["base_models"].pop(model_name)
    #             print(f'{model_name} was skiped')

    # train and predict
    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0 and options.debug:
            break
        # logger.debug(f'Training with fold {fold} started (train:{len(trn_idx)}, val:{len(val_idx)})')

        X_train = train.loc[trn_idx, :]
        X_valid = train.loc[val_idx, :]
        y_train = y_train_all.loc[trn_idx]
        y_valid = y_train_all.loc[val_idx]

        # modelの実行
        preds_t, preds_v = train_and_predict(
            X_train, X_valid, y_train, y_valid, test, CFG, ensemble=False
        )
        preds_v["val_idx"] = val_idx
        # 予測値の保存 
        preds_test = pd.concat([preds_test, preds_t], axis=0)
        preds_valid = pd.concat([preds_valid, preds_v], axis=0)

    if options.debug:  
        return
    
    preds_test = preds_test.groupby(level=0).mean() # indexが等しいもので平均
    preds_valid = preds_valid.sort_values('val_idx').reset_index(drop=True) # val_idxでソートし順番を元に戻す
    preds_valid = preds_valid.drop("val_idx", axis=1)

    # アンサンブル
    preds_valid.loc[:, "mean"] = preds_valid.agg("mean", axis=1)
    preds_test.loc[:, "mean"] = preds_test.agg("mean", axis=1)

    # 予測値の保存
    for name in preds_valid.columns:
        preds_valid.loc[:, [name]].to_feather(f'./data/output/{config_filename}/{name}_train.feather')
        preds_test.loc[:, [name]].to_feather(f'./data/output/{config_filename}/{name}_test.feather')
    
    # CVスコア
    cv_scores = {}
    for model_name in preds_valid.columns:
        score = mean_squared_error(y_train_all, preds_valid.loc[:, model_name], squared=False)
        score = '{:.5f}'.format(score)
        logger.debug('===CV scores===')
        logger.debug("model name : " + model_name)
        logger.debug("rmse score : " + str(score))
        cv_scores[model_name] = score

    # submitファイルの作成
    ID_name = CFG['ID_name']
    for sub_name in preds_test.columns:
        sub = pd.DataFrame(pd.read_csv('./data/input/sample_submission.csv')[ID_name])
        sub[CFG["target_name"]] = preds_test[sub_name]
        sub.to_csv(
            './data/output/{1}/sub_{0}_{2}.csv'.format(sub_name, config_filename, str(cv_scores[sub_name])),
            index=False
        )

if __name__ == '__main__':
    main()
    