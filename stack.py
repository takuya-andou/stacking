import os
import pickle
import numpy as np
import pandas as pd
from functools import reduce
import scipy.stats as stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss
from logging import getLogger, StreamHandler, DEBUG, Formatter
import copy
from collections import OrderedDict

# setting logger
logger = getLogger(__name__)
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(handler_format)
logger.addHandler(handler)

# settingSet the directory for trained models
fitted_models_dir = 'fitted_models'
if not os.path.isdir(fitted_models_dir):
    os.makedirs(fitted_models_dir)


# save model to pickle
def save_pkl(dir_name, filename, target_file):
    f = open(f'{dir_name}/{filename}.pkl', 'wb')
    pickle.dump(target_file, f)
    f.close


# load model from pickle
def load_pkl(dir_name, filename):
    f = open(f'{dir_name}/{filename}.pkl', 'rb')
    return pickle.load(f)


class StackModel:
    def __init__(self, model, model_name, x_names=None, regression=True, predict_proba=False, metric=None, k_fold=5, kfold_seed=0):
        self.model_name = model_name  # this model name
        self.x_names = x_names  # predictor variable names
        self.regression = regression
        self.predict_proba = predict_proba  # flg for predict probability
        self.metric = metric
        self.k_fold = k_fold  # k-fold cross-validation
        self.kfold_seed = kfold_seed  # random seed used for cross-validation
        self.models = []  # instance list of this model
        self.S_train = None  # predicted values for train data
        self.S_test = None  # predicted values for test data
        self.kf = None
        self.train_scores = []

        for i in range(k_fold):
            self.models.append(copy.deepcopy(model))

        if self.regression and self.predict_proba:
            self.regression = False

        if self.metric is None:
            if self.regression:
                self.metric = mean_absolute_error
            else:
                if self.predict_proba:
                    self.metric = log_loss
                else:
                    self.metric = accuracy_score

    def fit(self, X, y, refit=False):
        if os.path.exists(f'{fitted_models_dir}/{self.model_name}_models.pkl') == False or refit == True:
            logger.info(self.model_name + ' start fit')

            if self.x_names is None:
                self.x_names = X.columns.values.tolist()

            self.kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=self.kfold_seed).split(X, y)

            if self.regression:
                n_classes = 1
                dtype = float
            elif self.predict_proba:
                n_classes = len(np.unique(y))
                dtype = float
                y = y.astype(str)
            else:
                n_classes = 1
                dtype = object
                y = y.astype(str)

            S_train = np.zeros((X.shape[0], n_classes), dtype=dtype)

            for model, (tr_index, ts_index) in zip(self.models, self.kf):
                tr_x = X.iloc[tr_index][self.x_names]
                ts_x = X.iloc[ts_index][self.x_names]
                tr_y = y.iloc[tr_index]
                ts_y = y.iloc[ts_index]

                model.fit(tr_x, tr_y)
                if self.predict_proba:
                    tmp_pred = model.predict_proba(ts_x)
                else:
                    tmp_pred = model.predict(ts_x)
                self.train_scores.append(self.metric(ts_y, tmp_pred))

                S_train[ts_index, :] = tmp_pred.reshape(-1, n_classes)

            if not self.regression and self.predict_proba:
                self.S_train = pd.DataFrame(S_train, columns=[self.model_name + "_" + i for i in self.models[0].classes_])
            else:
                self.S_train = pd.Series(data=S_train.reshape(-1), index=X.index, name=self.model_name)

            logger.info(self.model_name + ' end fit')
            self.save_fit()

        else:
            self.load_fit()

    def predict(self, X, repredict=False):
        if os.path.exists(f'{fitted_models_dir}/{self.model_name}_S_test.pkl') == False or repredict == True:
            logger.info(self.model_name + ' start predict')

            S_test = []
            for model in self.models:
                if self.predict_proba:
                    S_test.append(model.predict_proba(X[self.x_names]))
                else:
                    S_test.append(model.predict(X[self.x_names]))
            S_test = np.stack(S_test)

            if self.regression or self.predict_proba:
                S_test = S_test.mean(axis=0)
            else:
                S_test = stats.mode(S_test, axis=0).mode
                S_test = S_test.reshape(S_test.shape[1:S_test.ndim])

            if not self.regression and self.predict_proba:
                self.S_test = pd.DataFrame(data=S_test, index=X.index, columns=[self.model_name + "_" + i for i in self.models[0].classes_])
            else:
                self.S_test = pd.Series(data=S_test, index=X.index, name=self.model_name)

            logger.info(self.model_name + ' end predict')
            self.save_predict()
        else:
            self.load_predict()

    def evaluate(self, y):
        return (self.metric(y, self.S_test))

    def save_fit(self):
        save_pkl(fitted_models_dir, self.model_name + '_models', self.models)
        save_pkl(fitted_models_dir, self.model_name + '_S_train', self.S_train)
        logger.info(self.model_name + ' save fit pkl')

    def save_predict(self):
        save_pkl(fitted_models_dir, self.model_name + '_S_test', self.S_test)
        logger.info(self.model_name + ' save pred pkl')

    def load_fit(self):
        self.models = load_pkl(fitted_models_dir, self.model_name + '_models')
        self.S_train = load_pkl(fitted_models_dir, self.model_name + '_S_train')
        logger.info(self.model_name + " load fit pkl")

    def load_predict(self):
        self.S_test = load_pkl(fitted_models_dir, self.model_name + '_S_test')
        logger.info(self.model_name + " load pred pkl")


class StackMaster:
    def __init__(self, models, merge_data=False):
        self.S_train = None  # predicted values for train data
        self.S_test = None  # predicted values for test data
        self.merge_data = merge_data
        tmp_models = OrderedDict()
        if type(models) == list:
            for model in models:
                tmp_models[model.model_name] = model
                self.models = tmp_models  # models dict
        else:
            self.models = models

    def fit(self, X, y, refit=False):
        S_train = []
        for model_name, model in self.models.items():
            model.fit(X, y, refit=refit)
            S_train.append(model.S_train)
        if len(self.models) >= 2:
            self.S_train = reduce(lambda left, right: pd.concat([left, right], axis=1, join='inner'), S_train)
        else:
            self.S_train = S_train[0]
        if self.merge_data:
            self.S_train = pd.merge(X, self.S_train, left_index=True, right_index=True, how='inner')

    def predict(self, X, repredict=False):
        S_test = []
        for model_name, model in self.models.items():
            model.predict(X, repredict=repredict)
            S_test.append(model.S_test)
        if len(self.models) >= 2:
            self.S_test = reduce(lambda left, right: pd.concat([left, right], axis=1, join='inner'), S_test)
        else:
            self.S_test = S_test[0]
        if self.merge_data:
            self.S_test = pd.merge(X, self.S_test, left_index=True, right_index=True, how='inner')
