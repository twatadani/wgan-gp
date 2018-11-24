# -*- coding: utf-8 -*-

##### config.py #####
# WGAN-GP用のコンフィグファイル
#
# created 2018/10/20 Takeyuki Watadani @ UT Radiology
##########
import numpy as np

# このセッションのデスクリプション
DESC = 'WGAN-GP-ver1'
DESC += '-04-lungHRCT-lambda5-deeplearningbox'

# 結果格納用ディレクトリ
RESULTDIR = './results/'

# データセット格納用のパス
DATASET_PATH = './datasets/'

# 教師用データセットとなる画像を保存しているパス
TRAIN_DATA_PATH = '/Users/watadani/OneDrive/src/lungHRCT_jpg/'

# 画像の拡張子
TRAIN_DATA_EXT = '.jpg'

# 教師用データセットを作成する画像数
TRAIN_DATA_NUMBER = 20000

# 今回用いる画像のピクセルサイズ
PIXELSIZE = 64

# 何ミニバッチごとにサンプルを保存するか
SAVE_SAMPLE_MINIBATCH = 300

# 教師用 保存データセットのプレフィクス
TRAIN_PREFIX = 'HRCT32'

# 各レイヤーでのバイアス使用の有無
USE_BIAS = False

# NNのドロップアウト率
DROPOUT_RATE = 0.5

# 学習のミニバッチサイズ
MINIBATCHSIZE = 32

# 学習のmax epoch
MAXEPOCH = 10000

# latent vectorの次元数
LATENT_VECTOR_SIZE = 128

# 学習が偏ったときにGとDの片方だけを集中学習するかどうか 基本はFalse
USE_INTENSIVE_TRAINING = False

# 学習率
C_LEARNING_RATE = 1e-4
G_LEARNING_RATE = 1e-4
BETA_1 = 0.5

# スレッドのスリープ間隔 (通常は変更不要)
SLEEP_INTERVAL = 0.8

# 乱数のseed
import time
SEED = round(time.time())

# TFRecord用の設定
# 1つのTFRecordファイルに格納するデータ数
TFRECORD_CAPACITY = 200

# TF Dataset API用の設定
# mapのparallel call数
DATASET_PARALLEL_CALLS = 4

# チェックポイントが残っていた場合に学習を再開するか
RESTART_TRAINING = True

# WGAN-GP用の設定
# lambda: 論文では10 
LAMBDA = 5

# ログ用の設定
import logging
import os.path
from datetime import datetime
LOGGER = logging.getLogger(DESC)
LOGGER.setLevel(10)
logdir = os.path.join(RESULTDIR, DESC)
if not os.path.exists(logdir):
    os.makedirs(logdir)
logfilename = 'log-'
now = datetime.now()
logfilename += str(now.year) + str(now.month) + str(now.day) + '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + '.txt'
logfile = os.path.join(logdir, logfilename)
if os.path.exists(logfile):
    os.remove(logfile)
fhandler = logging.FileHandler(logfile)
LOGGER.addHandler(fhandler)
shandler = logging.StreamHandler()
LOGGER.addHandler(shandler)


