import numpy as np
import pandas as pd
from stack import StackModel,StackMaster

# サンプルデータ用
from sklearn import datasets

# モデル
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
import xgboost as xgb

# サンプルデータの準備
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target_names[iris.target]
np.random.seed(0)
msk = np.random.choice(range(len(df)), 120, replace=False)
train_df = df.iloc[msk,]
test_df_x = df.drop(msk, axis=0).drop('petal width (cm)', axis=1)  # テストデータの説明変数(目的変数はdropしておく)
test_df_y = df.drop(msk, axis=0)['petal width (cm)']  # test_yは一番最後の相関係数出すときまで使わない！
print(train_df.head())
print(test_df_x.head())

lr1 = StackModel(
    model_name='lr1',
    model=LinearRegression,
    x_names=['sepal length (cm)', 'sepal width (cm)'],
    y_names='petal width (cm)')
lr2 = StackModel(
    model_name='lr2',
    model=LinearRegression,
    x_names=['petal length (cm)'],
    y_names='petal width (cm)',
    merge_method='mean',
    k_fold=3)
lr3 = StackModel(
    model_name='lr3',
    model=LinearRegression,
    x_names=['petal length (cm)'],
    y_names='petal width (cm)',
    merge_method='median',
    k_fold=3)
ri1 = StackModel(
    model_name='ri1',
    model=Ridge,
    x_names=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)'],
    y_names='petal width (cm)')
# 目的変数がラベルでクラス分類するならmerge_method='mode'(最頻値)にする
lda1 = StackModel(
    model_name='lda1',
    model=LinearDiscriminantAnalysis,
    x_names=['sepal length (cm)', 'sepal width (cm)'],
    y_names='species',
    merge_method='mode')
xgb1 = StackModel(
    model_name='xgb1',
    model=xgb.XGBRegressor,
    x_names=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)'],
    y_names='petal width (cm)')
# モデルのパラメータはこうやって指定する
xgb2 = StackModel(
    model_name='xgb2',
    model=xgb.XGBRegressor,
    x_names=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)'],
    y_names='petal width (cm)',
    params={'max_depth': 1})

master = StackMaster(
    models=[lr1, lr2, lr3, ri1, lda1, xgb1, xgb2])
master.fit(train_df, refit=False)
master.predict(test_df_x, repredict=False)

second1 = StackModel(
    model_name='second1',
    model=xgb.XGBRegressor,
    x_names=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'species_versicolor', 'species_virginica', 'lr1', 'lr2', 'lr3', 'ri1', 'lda1_versicolor', 'lda1_virginica', 'xgb1', 'xgb2'],
    y_names='petal width (cm)')

train_df_2 = pd.get_dummies(master.train_pred, drop_first=True)
test_df_x_2 = pd.get_dummies(master.test_pred, drop_first=True)

second1.fit(train_df_2, refit=True)
second1.predict(test_df_x_2, repredict=True)

print("スタッキング")
print(np.corrcoef(second1.test_pred, test_df_y)[0, 1])
print("ベース 線形回帰1")
print(np.corrcoef(lr1.test_pred, test_df_y)[0, 1])
print("ベース 線形回帰2")
print(np.corrcoef(lr2.test_pred, test_df_y)[0, 1])
print("ベース 線形回帰3")
print(np.corrcoef(lr3.test_pred, test_df_y)[0, 1])
print("ベース リッジ回帰")
print(np.corrcoef(ri1.test_pred, test_df_y)[0, 1])
print("ベース XGB1")
print(np.corrcoef(xgb1.test_pred, test_df_y)[0, 1])
print("ベース XGB2")
print(np.corrcoef(xgb2.test_pred, test_df_y)[0, 1])
