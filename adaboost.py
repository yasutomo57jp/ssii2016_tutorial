# -*- coding: utf-8 -*-

# from sklearn.datasets import fetch_mldata
# from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier as Classifier
from sklearn.metrics import confusion_matrix, accuracy_score
import time

# 比較をしやすくするため，予めtrain/testを分けたものを読み込む
from sklearn.externals import joblib
data_train, data_test, label_train, label_test = joblib.load("mnist")


# mnist = fetch_mldata("MNIST original", data_home=".")
# data = np.asarray(mnist.data, np.float32)
# data_train, data_test, label_train, label_test = train_test_split(data, mnist.target, test_size=0.2)

classifier = Classifier()
start = time.time()  # 処理時間の計測開始
classifier.fit(data_train, label_train)
training_time = time.time() - start

start = time.time()  # 処理時間の計測開始
result = classifier.predict(data_test)
predict_time = time.time() - start

# Confusion matrixを計算
print(training_time, predict_time)
cmat = confusion_matrix(label_test, result)
acc = accuracy_score(label_test, result)
print(cmat)
print(acc)
