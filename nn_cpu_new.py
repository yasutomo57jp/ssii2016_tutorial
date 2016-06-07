# -*- coding: utf-8 -*-

# Chainerの新しい書き方に対応したバージョン

import time
from chainer import Chain, Variable, optimizers
import chainer.links as L
import chainer.functions as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib
import numpy as np

# 学習データとテストデータに分ける
data_train, data_test, label_train, label_test = joblib.load("mnist")
data_train = np.asarray(data_train, np.float32)
data_test = np.asarray(data_test, np.float32)
label_train = np.asarray(label_train, np.int32)
label_test = np.asarray(label_test, np.int32)


# ネットワークのモデル
class MyNN(Chain):
    # 層のパラメータ
    def __init__(self):
        super(MyNN, self).__init__(
            l1=L.Linear(784, 200),
            l2=L.Linear(200, 100),
            l3=L.Linear(100, 10))
        self.is_train = True

    # 伝播のさせかた
    def __call__(self, x):
        h1 = F.dropout(F.relu(self.l1(x)), train=self.is_train)
        h2 = F.dropout(F.relu(self.l2(h1)), train=self.is_train)
        p = self.l3(h2)
        return p

network = MyNN()
model = L.Classifier(network)
model.compute_accuracy = True

# 学習のさせかた
optimizer = optimizers.Adam()
optimizer.setup(model)

# 学習
n_epoch = 100  # 学習繰り返し回数
batchsize = 20  # 学習データの分割サイズ
N = len(data_train)
losses = []  # 各回での誤差の変化を記録するための配列

start = time.time()  # 処理時間の計測開始
for epoch in range(n_epoch):
    print('epoch: %d' % (epoch+1))
    perm = np.random.permutation(N)  # 分割をランダムにするための並べ替え
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, batchsize):
        # 並べ替えた i〜i+batchsize 番目までのデータを使って学習
        x_batch = data_train[perm[i:i+batchsize]]
        t_batch = label_train[perm[i:i+batchsize]]

        # 初期化
        optimizer.zero_grads()

        # 順伝播
        x = Variable(x_batch)
        t = Variable(t_batch)
        loss = model(x, t)

        # 誤差逆伝播
        loss.backward()

        # 認識率の計算（表示用）
        accuracy = model.accuracy

        # パラメータ更新
        optimizer.update()

        sum_loss += float(loss.data) * batchsize
        sum_accuracy += float(accuracy.data) * batchsize

    losses.append(sum_loss / N)
    print("loss: %f, accuracy: %f" % (sum_loss / N, sum_accuracy / N))

training_time = time.time() - start
joblib.dump((model, training_time, losses), "classifiers/"+"nn_cpu")


# 評価
start = time.time()
x_test = Variable(data_test)
network.is_train = False
result_scores = network(x_test).data
predict_time = time.time() - start
results = np.argmax(result_scores, axis=1)

# %%
# 認識率を計算
score = accuracy_score(label_test, results)
print(training_time, predict_time)
print(score)
cmatrix = confusion_matrix(label_test, results)
print(cmatrix)
joblib.dump((training_time, predict_time, score, cmatrix), "results/"+"nn_cpu")

