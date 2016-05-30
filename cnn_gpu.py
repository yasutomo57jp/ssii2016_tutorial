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


# 学習データを画像に変換
def conv_feat_2_image(feats):
    data = np.ndarray((len(feats), 1, 28, 28), dtype=np.float32)
    for i, f in enumerate(feats):
        data[i] = f.reshape(28, 28)
    return data

data_train = conv_feat_2_image(data_train)
data_test = conv_feat_2_image(data_test)

# 層のパラメータ
model = FunctionSet(
            conv1=F.Convolution2D(1, 32, 3),
            conv2=F.Convolution2D(32, 64, 3),
            l1=F.Linear(576, 200),
            l2=F.Linear(200, 100),
            l3=F.Linear(100, 10)).to_gpu()


# 伝播のさせかた
def forward(x, is_train=True):
    h1 = F.max_pooling_2d(F.relu(model.conv1(x)), 3)
    h2 = F.max_pooling_2d(F.relu(model.conv2(h1)), 3)
    h3 = F.dropout(F.relu(model.l1(h2)), train=is_train)
    h4 = F.dropout(F.relu(model.l2(h3)), train=is_train)
    p = model.l3(h4)
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
joblib.dump((model, training_time, losses), "classifiers/"+"nn_cpu")


# 評価
start = time.time()
x_test = Variable(cuda.to_gpu(data_test))
result_scores = cuda.to_cpu(forward(x_test, is_train=False).data)
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
