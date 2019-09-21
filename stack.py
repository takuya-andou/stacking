import os
import pickle
import numpy as np
import pandas as pd
from functools import reduce
import scipy.stats as stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss
from logging import getLogger, StreamHandler, DEBUG, Formatter

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
    def __init__(self, model, model_name, x_names=None, regression=True, predict_proba=False, metric=None, k_fold=5, kfold_seed=0, params={}):
        self.model = model  # class of this model
        self.model_name = model_name  # this model name
        self.x_names = x_names  # predictor variable names
        self.regression = regression
        self.predict_proba = predict_proba  # flg for predict probability
        self.metric = metric
        self.k_fold = k_fold  # k-fold cross-validation
        self.kfold_seed = kfold_seed  # random seed used for cross-validation
        self.params = params  # hyper-parameter
        self.models = []  # instance list of this model
        self.train_pred = None  # predicted values for train data
        self.test_pred = None  # predicted values for test data
        self.kf = None
        self.train_scores = []

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

            if not self.regression and self.predict_proba:
                n_classes = len(np.unique(y))
                dtype = str
            else:
                n_classes = 1
                dtype = float

            train_pred = np.zeros((X.shape[0], n_classes), dtype=float)

            for fold_counter, (tr_index, ts_index) in enumerate(self.kf):  # k_fold回繰り返される
                tr_x = X.iloc[tr_index][self.x_names]
                ts_x = X.iloc[ts_index][self.x_names]
                tr_y = y.iloc[tr_index]
                ts_y = y.iloc[ts_index]

                model = self.model(**self.params)
                model.fit(tr_x, tr_y)
                if self.predict_proba:
                    tmp_pred = model.predict_proba(ts_x)
                else:
                    tmp_pred = model.predict(ts_x)
                self.train_scores.append(self.metric(ts_y, tmp_pred))

                train_pred[ts_index, :] = tmp_pred.reshape(-1, n_classes)
                self.models.append(model)

            if not self.regression and self.predict_proba:
                self.train_pred = pd.DataFrame(train_pred, columns=[self.model_name + "_" + i for i in self.models[0].classes_])
            else:
                self.train_pred = pd.Series(data=train_pred.reshape(-1), index=X.index, name=self.model_name)

            logger.info(self.model_name + ' end fit')
            self.save_fit()

        else:
            self.load_fit()

    def predict(self, X, repredict=False):
        if os.path.exists(f'{fitted_models_dir}/{self.model_name}_test_pred.pkl') == False or repredict == True:
            logger.info(self.model_name + ' start predict')

            test_pred = []
            for i, model in enumerate(self.models):
                if self.predict_proba:
                    test_pred.append(model.predict_proba(X[self.x_names]))
                else:
                    test_pred.append(model.predict(X[self.x_names]))
            test_pred = np.stack(test_pred)

            if self.regression or self.predict_proba:
                test_pred = test_pred.mean(axis=0)
            else:
                test_pred = stats.mode(test_pred, axis=0).mode
                test_pred = test_pred.reshape(test_pred.shape[1:test_pred.ndim])

            if not self.regression and self.predict_proba:
                self.test_pred = pd.DataFrame(data=test_pred, index=X.index, columns=[self.model_name + "_" + i for i in self.models[0].classes_])
            else:
                self.test_pred = pd.Series(data=test_pred, index=X.index, name=self.model_name)

            logger.info(self.model_name + ' end predict')
            self.save_predict()
        else:
            self.load_predict()

    def evaluate(self, y):
        return (self.metric(y, self.test_pred))

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


class StackMaster:
    def __init__(self, models, merge_data=False):
        self.models = models  # models list used for fit or predict
        self.train_pred = None  # predicted values for train data
        self.test_pred = None  # predicted values for test data
        self.merge_data = merge_data

    def fit(self, X, y, refit=False):
        train_pred = []
        for i, model in enumerate(self.models):
            model.fit(X, y, refit=refit)
            train_pred.append(model.train_pred)
        if len(self.models) >= 2:
            self.train_pred = reduce(lambda left, right: pd.concat([left, right], axis=1, join='inner'), train_pred)
        else:
            self.train_pred = train_pred[0]
        if self.merge_data:
            self.train_pred = pd.merge(X, self.train_pred, left_index=True, right_index=True, how='inner')

    def predict(self, X, repredict=False):
        test_pred = []
        for i, model in enumerate(self.models):
            model.predict(X, repredict=repredict)
            test_pred.append(model.test_pred)
        if len(self.models) >= 2:
            self.test_pred = reduce(lambda left, right: pd.concat([left, right], axis=1, join='inner'), test_pred)
        else:
            self.test_pred = test_pred[0]
        if self.merge_data:
            self.test_pred = pd.merge(X, self.test_pred, left_index=True, right_index=True, how='inner')
