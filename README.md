# practical_stacking

[practical_stacking](https://github.com/takuya-andou/stacking) is python package made in consideration of the practical use, not contests such as Kaggle. All trained model and out-of-fold predictions is automatically saved for next time. And if you try to train again a saved model, it is loaded automatically. In addition, you can only predict newly geted test data without training again a saved model. Of course you can choose scikit-learn, XGboost, Keras and other for stacking. 

[practical_stacking](https://github.com/takuya-andou/stacking)はKaggleなどのコンテストではなく実用することを考えて作られたpythonのパッケージです。全ての学習済モデルとOOF予測は、自動的に保存されます。そしてもし、学習済のモデルを再び学習しようとした場合、そのモデルは自動的にロードされます。加えて、保存してあるモデルを再学習することなく、新しく取得したテストデータの予測だけをすることもできます。当然、スタッキングのモデルにはscikit-learnやxgboostやkeras等を自由に選択することができます。

# Description

[Stacking](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking) (sometimes called stacked generalization) learns several models and combines their predictions. [This blog](http://mlwave.com/kaggle-ensembling-guide/) is very helpful to understand stacking and ensemble learning.

[スタッキング](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking)はいくつかのモデルを学習して、それらの予測を組み合わせます。[このブログ](http://mlwave.com/kaggle-ensembling-guide/)がスタッキングやアンサンブル学習の参考にとてもなります。

# Requirement

- python 3.5+(don't know about version < 3.5, sorry)
- os
- pickle
- functools.reduce
- scipy.stats
- numpy
- pandas
- sklearn

# Installation

You can install stacking from PyPI:
```
pip install practical_stacking
```

# Usage

See working example:

 * [Regression](https://github.com/takuya-andou/stacking/blob/master/examples/01_regression.ipynb)
 * [Classification (with class labels)](https://github.com/takuya-andou/stacking/blob/master/examples/02_classification_with_class_labels.ipynb)
 * [Classification (with class probabilities)](https://github.com/takuya-andou/stacking/blob/master/examples/03_classification_with_class_probabilities.ipynb)

See basic usage:

```python
# Import practical_stacking
from pracstack import StackModel, StackMaster

# Get your data

# Initialize 1st level models
models = [
    StackModel(model_name='Li',model=LinearRegression),
    StackModel(model_name='Ridge',model=Ridge)]

# Initialize StackMaster
master = StackMaster(models=models)

# Fit 1st level models
master.fit(X_train, y_train)

# Predict 1st level models
master.predict(X_test)

# Get your stacked features
S_train = master.S_train
S_test = master.S_test

# Use 2nd level models with stacked features
```