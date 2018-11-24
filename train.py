#!/usr/bin/env python
# -*- coding: utf-8 -*-

########## train.py ##########
#
#
# WGAN-GP
# ネットワークの学習を行うメインプログラム
#
# created 2018/10/16 Takeyuki Watadani
#
########################################

import config as cf
import functions as f
import critic as c
import generator as g

import random
import os.path
import math
import traceback
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import numpy as np
from PIL import Image

logger = cf.LOGGER

class Trainer:
    '''GANの学習を行うためのクラス'''

    def __init__(self, global_step, critic, generator):

        self.executor = ThreadPoolExecutor()

        self.C = critic
        self.G = generator
        self.training = False
        self.global_step_tensor = global_step
        self.session = None
        self.chkpdir = os.path.join(cf.RESULTDIR, cf.DESC)
        self.saver = None

    def train(self):
        '''学習を行うメインルーチン
        ここからqueuingスレッドと学習を行う本スレッドに分岐する'''

        self.training = True
        try:
            #C, Gの計算グラフを定義
            self.G.define_graph()
            self.C.define_graph()
            self.G.define_graph_postC()

            self.train_loop()
        except:
            import traceback
            traceback.print_exc()
        self.training = False
        return

    def save_sample_img(self, session, 
                        gstep, save_generator_img = True):
        '''Generatorが産生するサンプルイメージを保存する'''
        try:
            total_img = gstep * cf.MINIBATCHSIZE

            if save_generator_img:
                output_batch = session.run(self.G.output)
            else:
                output_batch = self.C.create_minibatch(session)

            # 新しいImageを作る
            sampleimg = Image.new(mode='L', size=(cf.PIXELSIZE * 3, cf.PIXELSIZE * 2))

            for i in range(6):
                if cf.MINIBATCHSIZE >= i:
                    img_np = output_batch[i].reshape((cf.PIXELSIZE, cf.PIXELSIZE))
                    img_uint8np = np.uint8(np.clip(img_np, 0.0, 255.0))
                    img_img = Image.fromarray(img_uint8np, mode='L')

                    sampleimg.paste(img_img, box=(cf.PIXELSIZE * (i % 3), cf.PIXELSIZE * (i // 3)))

            #save
            imgdir = os.path.join(cf.RESULTDIR, cf.DESC)
            interfix = '' if save_generator_img else 'C-'
            filename = os.path.join(imgdir, 'sample-' + interfix + str(total_img) + '.png')
            sampleimg.save(filename)
            prefix = 'G' if save_generator_img else 'C'
            logger.info(prefix + '由来のサンプル画像を保存しました。')
        except:
            traceback.print_exc()
                    

    def train_loop(self):
        '''学習を行うメインループを記述'''

        last_Dscore = 0.5

        try:
            merged = tf.summary.merge_all()
            self.saver = tf.train.Saver()

            with tf.Session() as self.session:

                self.summarywriter = tf.summary.FileWriter(self.chkpdir,
                                                           graph=self.session.graph)
                                                           # r1.8ではsessionなし
                                                           #,
                                                           #session=self.session)

                finished = False

                # 最新チェックポイントを読み込み、再開するかどうか決定する
                last_checkpoint = tf.train.latest_checkpoint(self.chkpdir)
                if (last_checkpoint is not None) and cf.RESTART_TRAINING == True:
                    try:
                        self.saver.restore(self.session, last_checkpoint)
                        logger.info('チェックポイントの復元に成功しました: ' + str(last_checkpoint))
                    except:
                        traceback.print_exc()
                        logger.info('チェックポイントの復元に失敗しました')
                        finished = True

                self.session.run([tf.global_variables_initializer(),
                                  tf.local_variables_initializer()])


                while not finished:
                    
                    gstep = self.session.run(self.global_step_tensor) // 2 

                    # Criticの学習
                    # WGANではCriticを多めに学習する
                    for __ in range(5):
                        _, last_Dscore, dloss = self.session.run(
                            [self.C.train_op, self.G.mean_C_score,
                             self.C.loss])

                    # Generatorの学習
                    _, WDistance, gloss = self.session.run([self.G.train_op,
                                                              self.G.mean_C_score,
                                                              self.G.loss])

                    current_epoch = gstep * cf.MINIBATCHSIZE / (cf.TRAIN_DATA_NUMBER * 6)

                    # 一定回数ごとに現状を表示
                    if gstep % 10 == 0:
                        total_img = gstep * cf.MINIBATCHSIZE
                        logger.info(
                            'Epoch: %.3f, total_img: %s, WDistance: %s, dloss: %s, gloss: %s',
                            current_epoch, total_img, WDistance, dloss, gloss)
                        try:
                            self.summarywriter.flush()
                        except:
                            traceback.print_exc()
                    # テスト: 一定回数毎にGのカーネル変数をチェック
                    if gstep % 25 == 0:
                        k4 = self.session.run(self.G.kernel4)#, feed_dict=fd)
                        logger.info(
                            'G_convkernel 4: mean=' + str(np.mean(k4)) + ', std=' + str(np.std(k4)))
                    # テスト: 一定回数毎にGのgradientをチェック
                    if gstep % 500 == 12:
                        gradients = self.session.run(self.G.gradients)#, feed_dict=fd)
                        for g in gradients:
                            mean = np.mean(g[0])
                            std = np.std(g[0])
                            logger.info('gradient mean: ' + str(mean) + ', gradient std: ' + str(std))

                    # 一定回数ごとにチェックポイントをセーブ
                    if gstep % 500 == 0:
                        try:
                            self.saver.save(self.session,
                                            os.path.join(self.chkpdir,'model.ckpt'),
                                            gstep)
                            logger.info('checkpointをセーブしました。')
                            summary = self.session.run(merged)
                            self.summarywriter.add_summary(summary, gstep)
                        except:
                            traceback.print_exc()

                    # 一定回数ごとにサンプル画像を保存
                    if gstep % cf.SAVE_SAMPLE_MINIBATCH == 0:
                        self.executor.submit(self.save_sample_img, self.session, gstep)
                    # 一定回数ごとに正解画像を保存
                    if gstep % 1000 ==0:
                        self.executor.submit(self.save_sample_img, self.session, gstep, False)

                    # 最大epochに到達したら終了フラグを立てる
                    if current_epoch >= cf.MAXEPOCH:
                        logger.info('max_epochに到達したため終了します。')
                        finished = True
        except:
            traceback.print_exc()
            self.training = False
        finally:
            logger.info('train_loopを終了します。')


logger.info('WGAN 学習プログラムを開始します。')

# 乱数を初期化
logger.info('乱数を初期化します')
random.seed()

# ネットワークのインスタンスを作成
global_step_tensor = tf.train.create_global_step()

C = c.Critic(global_step_tensor)
G = g.Generator(C, global_step_tensor)

T = Trainer(global_step_tensor, C, G)

logger.info('nn.trainを開始します。')
T.train()

logger.info('学習が終了しました。')
chkpdir = os.path.join(cf.RESULTDIR, cf.DESC)
chkp = tf.train.latest_checkpoint(chkpdir)
if chkpdir is not None and chkp is not None:
    logger.info('最新チェックポイント: ' + tf.train.latest_checkpoint(chkpdir))
logger.info('プログラムを終了します。')

