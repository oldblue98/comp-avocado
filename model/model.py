import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier as RF
from catboost import CatBoostClassifier as cat
# from .utils import Sampling

import torch
from torch import nn

from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline

from abc import abstractmethod

## loggerの設定
from logging import getLogger
logger =getLogger("logger")

pd.set_option('display.max_rows', None)


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, config, ensemble=False):
    preds_valid = pd.DataFrame()
    preds_test = pd.DataFrame()
    target_name = config["target_name"]

    # target-encoding用のtest作成
    cols = [c for c in X_train.columns if X_train.dtypes[c]=="object"]
    print(cols)
    
    # for c in cols:
    #     data_tmp = pd.DataFrame({c: X_train.loc[:, c], target_name: y_train.iloc[:, 0]})
    #     target_mean = data_tmp.groupby(c)[target_name].mean()
    #     # テストデータのカテゴリ変数を各平均ごとに置換
    #     X_test.loc[:, c] = X_test[c].map(target_mean)

    # resampling
    # X_train, y_train = Sampling(X_train, y_train, mode=config["mode"])

    # X_train, X_valid = target_encoding(X_train, X_valid, y_train, cols, target_name, config)


    if ensemble == 0:
        models = config['base_models']
    elif ensemble == 1:
        models = config['meta_models']

    for name, params in models.items():
        if name == "LightGBM":
            model = LightGBM()
        elif name == "Lasso":
            model = LassoWrapper()
        elif name == "Logistic":
            model = LogisticWrapper()
        elif name == "Ridge":
            model = RidgeWrapper()
        elif name == "ElasticNet":
            model = ElasticNetWrapper()
        elif name == "KernelRidge":
            model = KernelRidgeWrapper()
        elif name == "SVM":
            model = SVMWrapper()
        elif name == "SVR":
            model = SVRWrapper()
        elif name == "XGBoost":
            model = XGBoost()
        elif name == "RandomForest":
            model = RandomForestWrapper()
        elif name == "GradientBoosting":
            model = GradientBoostingRegressorWrapper()
        elif name == "CatBoost":
            model = CatBoost()
        else:
            continue

        pred_t, pred_v, m = model.train_and_predict(X_train, X_valid, y_train, y_valid, X_test, params)
        preds_valid[name] = pred_v
        preds_test[name] = pred_t

    return preds_test, preds_valid

class Model:
    @abstractmethod
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        raise NotImplementedError

# アンサンブルモデル
class RandomForestWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        reg = make_pipeline(RobustScaler(), RF(**params))
        reg.fit(X_train, y_train)

        y_valid_pred = reg.predict_proba(X_valid)[:, 1]
        y_pred = reg.predict_proba(X_test)[:, 1]
        return y_pred, y_valid_pred, reg

class GradientBoostingRegressorWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        reg = make_pipeline(RobustScaler(), GradientBoostingRegressor(**params))
        reg.fit(X_train, y_train)

        y_valid_pred = reg.predict(X_valid)
        y_pred = reg.predict(X_test)
        return y_pred, y_valid_pred, reg

class CatBoost(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        reg = make_pipeline(RobustScaler(), cat(**params))
        reg.fit(X_train, y_train)

        y_valid_pred = reg.predict_proba(X_valid)[:, 1]
        y_pred = reg.predict_proba(X_test)[:, 1]
        return y_pred, y_valid_pred, reg

class LightGBM(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        model = lgb.train(
            params, lgb_train,
            valid_sets=lgb_eval,
            num_boost_round=5000,
            early_stopping_rounds=100
        )
        pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
        pred_test = model.predict(X_test, num_iteration=model.best_iteration)

        importance = pd.DataFrame(model.feature_importance(importance_type = "gain"), index=X_train.columns, columns=['importance'])
        importance = importance.sort_values('importance', ascending=False)
        logger.debug("Light GBM importance (gain)")
        logger.debug(importance)
        importance = pd.DataFrame(model.feature_importance(importance_type = "split"), index=X_train.columns, columns=['importance'])
        importance = importance.sort_values('importance', ascending=False)
        logger.debug("Light GBM importance (split)")
        logger.debug(importance)
        return pred_test, pred_valid , model

class XGBoost(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):

        dtrain = xgb.DMatrix(X_train, y_train)
        deval = xgb.DMatrix(X_valid, y_valid)

        watchlist = [(deval, 'eval'), (dtrain, 'train')]


        model = xgb.train(
            params, dtrain,
            num_boost_round=5000,
            evals=watchlist,
            early_stopping_rounds=10,
            verbose_eval = False
        )

        y_valid_pred = model.predict(deval)
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)

        return y_pred, y_valid_pred, model


class KernelRidgeWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        reg = make_pipeline(RobustScaler(), KernelRidge(alpha=params['alpha'], kernel=params['kernel'], degree=params['degree'], coef0=params['coef0']))
        reg.fit(X_train, y_train)
        y_valid_pred = reg.predict(X_valid)
        y_pred = reg.predict(X_test)
        return y_pred, y_valid_pred, reg


class SVRWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        reg = make_pipeline(RobustScaler(), SVR(kernel=params['kernel'], degree=params['degree'], coef0=params['coef0'], C=params['C'], epsilon=params['epsilon']))
        reg.fit(X_train, y_train)
        y_valid_pred = reg.predict(X_valid)
        y_pred = reg.predict(X_test)
        return y_pred, y_valid_pred, reg


# Linear models

class LassoWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        lr = make_pipeline(RobustScaler(), Lasso(alpha=params['alpha']))
        lr.fit(X_train, y_train)
        y_valid_pred = lr.predict(X_valid)
        y_pred = lr.predict(X_test)
        return y_pred, y_valid_pred, lr


class RidgeWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        lr = make_pipeline(RobustScaler(), Ridge(alpha=params['alpha']))
        lr.fit(X_train, y_train)

        y_valid_pred = lr.predict(X_valid)
        y_pred = lr.predict(X_test)
        return y_pred, y_valid_pred, lr


class ElasticNetWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        lr = make_pipeline(RobustScaler(), ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio']))
        lr.fit(X_train, y_train)

        y_valid_pred = lr.predict(X_valid)
        y_pred = lr.predict(X_test)
        return y_pred, y_valid_pred, lr

        
class LogisticWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        lr = make_pipeline(RobustScaler(), LogisticRegression())
        lr.fit(X_train, y_train)
        y_valid_pred = lr.predict_proba(X_valid)[:, 1]
        y_pred = lr.predict_proba(X_test)[:, 1]
        return y_pred, y_valid_pred, lr

class SVMWrapper(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        lr = make_pipeline(RobustScaler(), SVC(params))
        lr.fit(X_train, y_train)
        y_pred_val = lr.predict_proba(X_valid)
        y_pred_test = lr.predict_proba(X_test)
        return y_pred_test, y_pred_val, lr
'''
# Deep learning
class NeuralNetwork(Model):
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        # data scaling 
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
        
        y_train = np.asarray(y_train)
        y_valid = np.asarray(y_valid)

        # architecture
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1], )))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # model.add(Dense(32, activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        optimizer = optimizers.Adam(learning_rate = params["lr"])
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        batch_size = params["batch_size"]
        epochs = params["epochs"]
        
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_valid, y_valid))

        # predict
        y_valid_pred = model.predict(X_valid)[..., 0]
        y_pred = model.predict(X_test)[..., 0]
        
        return y_pred, y_valid_pred, model
'''

def target_encoding(X_train, X_valid, y_train, cols, target_name, config):
    for c in cols:
        data_tmp = pd.DataFrame({c: X_train.loc[:, c], target_name: y_train.iloc[:, 0]})
        target_mean = data_tmp.groupby(c)[target_name].mean()
        # バリデーションデータのカテゴリを置換
        X_valid.loc[:, c] = X_valid[c].map(target_mean)
        tmp = np.repeat(np.nan, X_train.shape[0])

        qcut_target = pd.qcut(y_train.iloc[:, 0], config["q_splits"], labels=False)
        kf_encoding = StratifiedKFold(n_splits=config['fold_num'], shuffle=True, random_state=config['seed']).split(np.arange(X_train.shape[0]), qcut_target)
        for tr_idx, va_idx in kf_encoding:
            # oofで各カテゴリにおける目的変数の平均を計算
            target_mean = data_tmp.iloc[tr_idx].groupby(c)[target_name].mean()
            # 変換後の値を一時配列に格納
            tmp[va_idx] = X_train[c].iloc[va_idx].map(target_mean)
        X_train.loc[:, c] = tmp
    # print("te : ", X_train.head())
    return X_train, X_valid
