import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import LabelEncoder

os.makedirs("./features/feather/train/", exist_ok=True)
os.makedirs("./features/feather/test/", exist_ok=True)

CFG = json.load(open("./configs/default.json"))

def create_features():
    train = pd.read_csv("./data/input/train.csv")
    test = pd.read_csv(("./data/input/test.csv"))
    n_train = train.shape[0]

    # 変数作成
    df = pd.concat([train, test], axis=0).reset_index(drop=True)

    # 月
    df["month"] = pd.to_datetime(df["Date"]).dt.month
    # df["month_name"] = pd.to_datetime(df["Date"]).dt.month.apply(lambda x: str(x))
    
    # 統計量
    # df["vol_year_mean"] = df.groupby("year")["Total Volume"].mean()
    # df["bags_year_mean"] = df.groupby("year")["Total Bags"].mean()
    # df["vol_year_std"] = df.groupby("year")["Total Volume"].std()
    # df["bags_year_std"] = df.groupby("year")["Total Bags"].std()

    # 比を計算
    df["s_rate"] = df["Small Bags"] / df["Total Bags"]
    df["l_rate"] = df["Large Bags"] / df["Total Bags"]
    df["xl_rate"] = df["XLarge Bags"] / df["Total Bags"]
    df["4046_rate"] = df["4046"] / df["Total Volume"]
    df["4225_rate"] = df["4225"] / df["Total Volume"]
    df["4770_rate"] = df["4770"] / df["Total Volume"]

    # 時間シフト　
    shift_num = 10
    interval = 1
    cols = [
        'Total Volume',
        '4046_rate', 
        '4225_rate', 
        '4770_rate',
        'Total Bags', 
        's_rate', 
        'l_rate', 
        'xl_rate',
    ]
    for name in cols:
        tmp = df.copy()
        df[f'{name}_avg10'] = tmp.sort_values("Date").groupby(["region", "type"])[name].shift(1).rolling(window=10).mean()
        df[f'{name}_avg5'] = tmp.sort_values("Date").groupby(["region", "type"])[name].shift(1).rolling(window=5).mean()
        df[f'{name}_max10'] = tmp.sort_values("Date").groupby(["region", "type"])[name].shift(1).rolling(window=10).max()
        df[f'{name}_max5'] = tmp.sort_values("Date").groupby(["region", "type"])[name].shift(1).rolling(window=5).max()
        for i in range(interval, interval*shift_num+1, interval):
            df[f'{name}_{i}'] = tmp.sort_values("Date").groupby(["region", "type"])[name].shift(i)
            df[f'{name}_{i}'] = tmp.sort_values("Date").groupby(["region", "type"])[name].shift(i)

    # ダミー変数化
    df = pd.concat([df, pd.get_dummies(df.loc[:, ["type", "region"]])], axis=1)
    
    # ラベルエンコード
    le = LabelEncoder()
    for column in ['type','region']:
        le = LabelEncoder()
        le.fit(df[column])
        df[f'{column}_label'] = le.transform(df[column])

    # train, testに分割
    train = df.iloc[:n_train, :]
    train = train.dropna(how='any', axis=0).reset_index(drop=True)
    print(train.shape)
    print(test.shape)
    test = df.iloc[n_train:, :].drop(CFG["target_name"], axis=1).reset_index(drop=True)

    print(train.shape)
    print(train.shape)
    for name in train.columns:
            train.loc[:, [name]].to_feather('./features/feather/train/{}.feather'.format(name))
            print('train/{}.feather created'.format(name))

    for name in test.columns:
        test.loc[:, [name]].to_feather('./features/feather/test/{}.feather'.format(name))
        print('test/{}.feather created'.format(name))
        
if __name__ == '__main__':
    create_features()
    