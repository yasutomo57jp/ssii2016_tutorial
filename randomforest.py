# -*- coding: utf-8 -*-
# from sklearn.datasets import fetch_mldata
# from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# 比較をしやすくするため，予めtrain/testを分けたものを読み込む
from sklearn.externals import joblib
data_train, data_test, label_train, label_test = joblib.load("mnist")

# mnist = fetch_mldata("MNIST original", data_home=".")
# data = np.asarray(mnist.data, np.float32)
# data_train, data_test, label_train, label_test = train_test_split(data, mnist.target, test_size=0.2)

classifier = RandomForestClassifier()
classifier.fit(data_train, label_train)

result = classifier.predict(data_test)

cmat = confusion_matrix(label_test, result)
print(cmat)
