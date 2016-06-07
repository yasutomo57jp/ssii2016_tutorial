#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import numpy as np


# データセットのダウンロードと読み込み
mnist = fetch_mldata("MNIST original", data_home=".")

# 特徴量の形式の変換
data = np.asarray(mnist.data, np.float64)

# 学習データとテストデータに分割（8:2）
dataset = train_test_split(data, mnist.target, test_size=0.2)

# "mnist" という名前で保存
joblib.dump(dataset, "mnist")

