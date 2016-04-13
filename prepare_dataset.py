#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import numpy as np


mnist = fetch_mldata("MNIST original", data_home=".")

data = np.asarray(mnist.data, np.float64)
dataset = train_test_split(data, mnist.target, test_size=0.2)

joblib.dump(dataset, "mnist")

