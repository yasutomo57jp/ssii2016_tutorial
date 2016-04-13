# SSII 2016 チュートリアル
## Pythonによる機械学習

SSII 2016において，[Pythonによる機械学習](https://confit.atlas.jp/guide/event/ssii2016/static/tutorial#TS22)というチュートリアル講演をします．
講演スライドは後日公開予定です．

ここでは説明に使ったサンプルコードを置いています．

## 使い方

* まず， prepare_dataset.py を実行して，データセットを用意してください．
* 次に，svm.py や randomforest.py，cnn_gpu.py などを実行すると，学習の後に識別スコア等が表示されます．

```bash
 python prepare_dataset.py
 python svm.py
 python cnn_gpu.py
```


## 必要環境

* Python 2.7
	* chainer がインストールしてあること
* nn_gpu.pyやcnn_gpu.py を実行する場合，CUDAをインストールした後にchainerをインストール
	* ただし，CUDAに対応したGPUが必要です

### CUDA インストール

[こちら](https://developer.nvidia.com/cuda-downloads)から自分の環境に合うものをダウンロードしてインストールしてください

### chainer インストール
* 以下のコマンドを実行

```bash
pip install chainer
```
