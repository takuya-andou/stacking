import os
import pickle
import numpy as np
import pandas as pd
from functools import reduce
import scipy.stats as stats
from sklearn.model_selection import KFold
from logging import getLogger, StreamHandler, DEBUG, Formatter

logger = getLogger(__name__)
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(handler_format)
logger.addHandler(handler)

# 学習済みモデルのディレクトリ指定
fitted_models_dir = 'fitted_models'
if not os.path.isdir(fitted_models_dir):
    os.makedirs(fitted_models_dir)


# pklに保存したリロードしたり
def save_pkl(dir_name, filename, target_file):
    f = open(f'{dir_name}/{filename}.pkl', 'wb')
    pickle.dump(target_file, f)
    f.close


# pklファイルからロード
def load_pkl(dir_name, filename):
    f = open(f'{dir_name}/{filename}.pkl', 'rb')
    return pickle.load(f)


class StackModel:
    def __init__(self, model, model_name, x_names=None, k_fold=5, kfold_seed=0, merge_method='mean', params={}):
        self.model_name = model_name  # モデル名(いらないかも?)
        self.x_names = x_names  # 説明変数のカラム名(リスト)
        self.k_fold = k_fold  # 学習時のk-foldの分割数
        self.train_pred = None  # 学習データの予測値
        self.test_pred = None  # テストデータの予測値
        self.model = model
        self.params = params
        self.models = []  # k_foldのときできる複数のモデルを格納
        self.kfold_seed = kfold_seed  # ランダムシード
        self.merge_method = merge_method  # 複数モデルの予測値をマージする方法

    def fit(self, X, y, refit=False):
        if os.path.exists(f'{fitted_models_dir}/{self.model_name}_models.pkl') == False or refit == True:  # 学習済みじゃないか、再学習しろと言われているか
            logger.info(self.model_name + ' start fit')

            if self.x_names is None:
                self.x_names = X.columns.values.tolist()

            kf = list(KFold(n_splits=self.k_fold, shuffle=True, random_state=self.kfold_seed).split(X))  # indexじゃなくて行数が返ってくる
            train_pred = np.empty(len(X), dtype=y.dtype)
            for i, (tr_index, ts_index) in enumerate(kf):  # k_fold回繰り返される
                tr_x = X.iloc[tr_index][self.x_names]
                ts_x = X.iloc[ts_index][self.x_names]
                tr_y = y.iloc[tr_index]
                model_ = self.model(**self.params)
                model_.fit(tr_x, tr_y)  # 学習 model()依存！！！
                train_pred[ts_index] = model_.predict(ts_x)  # 予測 model()依存！！！
                self.models.append(model_)  # 学習済みモデルに追加
            self.train_pred = pd.Series(data=train_pred, index=X.index)  # インスタンスにもたせておく
            logger.info(self.model_name + ' end fit')
            self.save_fit()
        else:
            self.load_fit()

    def predict(self, X, repredict=False):
        if os.path.exists(f'{fitted_models_dir}/{self.model_name}_test_pred.pkl') == False or repredict == True:  # 予測済みじゃないか、再予測しろと言われているか
            logger.info(self.model_name + ' start predict')

            predict = []
            for i, model_ in enumerate(self.models):  # モデル数分回る
                predict.append(model_.predict(X[self.x_names]))  # 予測する  model()依存！！！
            predict = np.stack(predict)

            if self.merge_method is 'mean':
                self.test_pred = pd.Series(data=predict.mean(axis=0), index=X.index)  # 各モデルの予測の平均値を取る
            elif self.merge_method is 'median':
                self.test_pred = pd.Series(data=np.median(predict, axis=0), index=X.index)  # 各モデルの予測の中央値を取る
            elif self.merge_method is 'mode':
                predict = stats.mode(predict, axis=0).mode  # 各モデルの最頻値を取る
                self.test_pred = pd.Series(data=predict.reshape(predict.shape[1:predict.ndim]), index=X.index)

            logger.info(self.model_name + ' end predict')
            self.save_predict()
        else:
            self.load_predict()

    def save_fit(self):
        save_pkl(fitted_models_dir, self.model_name + '_models', self.models)
        save_pkl(fitted_models_dir, self.model_name + '_train_pred', self.train_pred)
        logger.info(self.model_name + ' save fit pkl')

    def save_predict(self):
        save_pkl(fitted_models_dir, self.model_name + '_test_pred', self.test_pred)
        logger.info(self.model_name + ' save pred pkl')

    def load_fit(self):
        self.models = load_pkl(fitted_models_dir, self.model_name + '_models')
        self.train_pred = load_pkl(fitted_models_dir, self.model_name + '_train_pred')
        logger.info(self.model_name + " load fit pkl")

    def load_predict(self):
        self.test_pred = load_pkl(fitted_models_dir, self.model_name + '_test_pred')
        logger.info(self.model_name + " load pred pkl")


# 各モデルをいっぺんに学習させたり予測させたりするもの
class StackMaster:
    def __init__(self, models, merge_data=False):
        self.models = models  # 学習・予測に使うモデル名のリスト
        self.train_pred = None  # 学習データの予測値
        self.test_pred = None  # テストデータの予測値
        self.merge_data = merge_data

    def fit(self, X, y, refit=False):
        train_pred = []
        model_names = []
        for i, model in enumerate(self.models):
            model.fit(X, y, refit=refit)
            train_pred.append(model.train_pred)
            model_names.append(model.model_name)
        if len(self.models) >= 2:
            self.train_pred = reduce(lambda left, right: pd.concat([left, right], axis=1, join='inner'), train_pred)
            self.train_pred.columns = model_names
        else:
            self.train_pred = pd.DataFrame({model_names[0]: train_pred[0]})
        if self.merge_data:
            self.train_pred = pd.merge(X, self.train_pred, left_index=True, right_index=True, how='inner')

    def predict(self, X, repredict=False):
        test_pred = []
        model_names = []
        for i, model in enumerate(self.models):
            model.predict(X, repredict=repredict)
            test_pred.append(model.test_pred)
            model_names.append(model.model_name)
        if len(self.models) >= 2:
            self.test_pred = reduce(lambda left, right: pd.concat([left, right], axis=1, join='inner'), test_pred)
            self.test_pred.columns = model_names
        else:
            self.test_pred = pd.DataFrame({model_names[0]: test_pred[0]})
        if self.merge_data:
            self.test_pred = pd.merge(X, self.test_pred, left_index=True, right_index=True, how='inner')
