# -*- coding: utf-8 -*-

import time
from chainer import FunctionSet, Variable, optimizers, cuda
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

# 層のパラメータ
model = FunctionSet(
            l1=F.Linear(784, 200),
            l2=F.Linear(200, 100),
            l3=F.Linear(100, 10)).to_gpu()


# 伝播のさせかた
def forward(x, is_train=True):
    h1 = F.dropout(F.relu(model.l1(x)), train=is_train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=is_train)
    p = model.l3(h2)
    return p

# 学習のさせかた
optimizer = optimizers.Adam()
optimizer.setup(model)

# 学習
n_epoch = 100
batchsize = 20
N = len(data_train)
losses = []
start = time.time()
for epoch in range(n_epoch):
    print('epoch: %d' % (epoch+1))
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, batchsize):
        x_batch = data_train[perm[i:i+batchsize]]
        t_batch = label_train[perm[i:i+batchsize]]

        optimizer.zero_grads()

        x = Variable(cuda.to_gpu(x_batch))
        t = Variable(cuda.to_gpu(t_batch))
        p = forward(x)

        loss = F.softmax_cross_entropy(p, t)
        accuracy = F.accuracy(p, t)

        loss.backward()

        optimizer.update()

        sum_loss += float(loss.data) * batchsize
        sum_accuracy += float(accuracy.data) * batchsize

    losses.append(sum_loss / N)
    print("loss: %f, accuracy: %f" % (sum_loss / N, sum_accuracy / N))

training_time = time.time() - start
joblib.dump((model, training_time, losses), "classifiers/"+"nn_gpu")


# 評価
start = time.time()
x_test = Variable(cuda.to_gpu(data_test))
result_scores = forward(x_test, is_train=False).to_cpu()
predict_time = time.time() - start
results = np.argmax(result_scores, axis=1)

# %%
# 認識率を計算
score = accuracy_score(label_test, results)
print(training_time, predict_time)
print(score)
cmatrix = confusion_matrix(label_test, results)
print(cmatrix)
joblib.dump((training_time, predict_time, score, cmatrix), "results/"+"nn_gpu")
