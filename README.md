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

To install stacking, `cd` to the stacking folder and run the install command**(up-to-date version, recommended)**:
```
sudo python setup.py install
```

You can also install stacking from PyPI:
```
pip install stacking
```

# Usage

## StackModel

## StackMaster
