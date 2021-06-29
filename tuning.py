import argparse
import json
import os
import datetime
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb

from model.utils import load_table
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
from sklearn.metrics import mean_squared_error
from model.utils import load_table

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
parser.add_argument('--device', default='0')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--ensemble', action='store_true')
options = parser.parse_args()
CFG = json.load(open(options.config))
# device = torch.device(f'cuda:{options.device}')

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
handler2 = FileHandler(filename=f'./logs/{config_filename}_tune.log')
handler2.setLevel(DEBUG)
handler2.setFormatter(Formatter("%(asctime)s: %(message)s"))
#loggerに2つのハンドラを設定
logger.addHandler(handler1)
logger.addHandler(handler2)

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

for fold, (trn_idx, val_idx) in enumerate(folds):
    X_train = train.loc[trn_idx, :]
    X_valid = train.loc[val_idx, :]
    y_train = y_train_all.loc[trn_idx]
    y_valid = y_train_all.loc[val_idx]
    break

def main():
    logger.debug(CFG["columns"])
    if not options.ensemble:
        models = CFG['base_models']
    elif options.ensemble:
        models = CFG['meta_models']

    for name, params in models.items():
        if name == "LightGBM":
            objective = LightGBM
        elif name == "Lasso":
            objective = LassoWrapper
        else:
            continue
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        trial = study.best_trial

        logger.debug('Best Trial')
        logger.debug('\tValue: {}'.format(trial.value))
        logger.debug(' \tParams: ')
        for key, value in trial.params.items():
            logger.debug(name)
            logger.debug('\t\t{}: {}'.format(key, value))

def LightGBM(trial):
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid)

    param = {
            'objective' : 'regression',
            'metric': 'rmse',
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            "max_depth" : trial.suggest_int('max_depth', 2, 8),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 1e-8, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 1e-8, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

    model = lgb.train(
        param, lgb_train,
        valid_sets=lgb_valid,
        early_stopping_rounds=100
        )

    y_pred = model.predict(X_valid)
    score = mean_squared_error(y_valid, y_pred, squared=False)
    return score

def LassoWrapper(trial):
    params = {
        "alpha" : trial.suggest_uniform("alpha", 0., 10.0)
    }
    lr = make_pipeline(RobustScaler(), Lasso(alpha=params['alpha']))
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_valid)
    score = mean_squared_error(y_valid, y_pred, squared=False)
    return score

if __name__ == '__main__':
    main()