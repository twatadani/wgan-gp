# -*- coding: utf-8 -*-

########## critic.py ##########
#
# WGAN-GP critic
# 
#
# created 2018/10/20 Takeyuki Watadani @ UT Radiology
#
########################################

import config as cf
import functions as f

import tensorflow as tf
import numpy as np

import os.path
import io
import math
import random

logger = cf.LOGGER

class Critic:
    '''WGAN-GPのCriticを記述するクラス'''


    def __init__(self, global_step):
        self.global_step_tensor = global_step

        self.from_dataset = None # データセットからの入力画像
        self.from_generator = None # generatorからの入力画像
        self.output = None # 出力の確率
        self.optimizer = None
        self.train_op = None

    def define_forward(self, input, vreuse = None):
        '''判定する計算を返す'''
        
        with tf.variable_scope('C_network', reuse=vreuse):

            norm_factor = tf.constant(255.0, dtype=tf.float32,
                                      shape = (cf.MINIBATCHSIZE, cf.PIXELSIZE, cf.PIXELSIZE, 1))
            #inreshaped = tf.divide(tf.reshape(input,
            #                                  shape = (-1, cf.PIXELSIZE, cf.PIXELSIZE, 1),
            #                                  name = 'C_inreshaped'), norm_factor)
            inreshaped = tf.reshape(input,
                                    shape = (-1, cf.PIXELSIZE, cf.PIXELSIZE, 1),
                                    name = 'C_inreshaped')

            c1 = 'C_conv1'
            conv1 = f.apply_dobn(tf.layers.conv2d(inputs = inreshaped,
                                                  filters = 64,
                                                  kernel_size = (4, 4),
                                                  strides = (2, 2),
                                                  padding = 'same',
                                                  use_bias = cf.USE_BIAS,
                                                  kernel_initializer = tf.initializers.random_uniform(minval=-1.0, maxval=1.0),
                                                  activation = tf.nn.leaky_relu,
                                                  name = c1),
                                 c1)
            f.print_shape(conv1)

            c2 = 'C_conv2'
            conv2 = f.apply_dobn(tf.layers.conv2d(inputs = conv1,
                                                  filters = 128,
                                                  kernel_size = (4, 4),
                                                  strides = (2, 2),
                                                  padding = 'same',
                                                  use_bias = cf.USE_BIAS,
                                                  kernel_initializer = tf.initializers.random_uniform(minval=-1.0, maxval=1.0),
                                                  activation = tf.nn.leaky_relu,
                                                  name = c2),
                                 c2)
            f.print_shape(conv2)

            c3 = 'C_conv3'
            conv3 = f.apply_dobn(tf.layers.conv2d(inputs = conv2,
                                                  filters = 256,
                                                  kernel_size = (4, 4),
                                                  strides = (2, 2),
                                                  padding = 'same',
                                                  use_bias = cf.USE_BIAS,
                                                  kernel_initializer = tf.initializers.random_uniform(minval=-1.0, maxval=1.0),
                                                  activation = tf.nn.leaky_relu,
                                                  name = c3),
                                 c3)
            f.print_shape(conv3)

            c4 = 'C_conv4'
            conv4 = f.apply_dobn(tf.layers.conv2d(inputs = conv3,
                                                  filters = 512,
                                                  kernel_size = (4, 4),
                                                  strides = (2, 2),
                                                  padding = 'same',
                                                  use_bias = cf.USE_BIAS,
                                                  kernel_initializer = tf.initializers.random_uniform(minval=-1.0, maxval=1.0),
                                                  activation = tf.nn.leaky_relu,
                                                  name = c4),
                                 c4)
            f.print_shape(conv4)

            flatten = tf.layers.flatten(conv4,
                                        name = 'C_flatten')
            f.print_shape(flatten)

            cfinal = 'C_convfinal'
            convfinal = f.apply_dobn(tf.layers.conv2d(inputs = conv4,
                                                      filters = 1,
                                                      kernel_size = (4, 4),
                                                      strides = (1, 1),
                                                      padding = 'valid',
                                                      use_bias = cf.USE_BIAS,
                                                      kernel_initializer = tf.initializers.random_uniform(minval=-1.0, maxval=1.0),
                                                      activation = tf.nn.leaky_relu,
                                                      name = cfinal),
                                     cfinal)
            f.print_shape(convfinal)

            return tf.reshape(convfinal,
                              shape = (-1, 1))

    def define_graph(self):
        '''discriminatorの計算グラフを定義する'''

        with tf.variable_scope('C_network'):

            self.from_dataset = f.obtain_minibatch_op()
            print(str(self.from_dataset))
        
            self.p_fake = self.define_forward(self.from_generator, vreuse=tf.AUTO_REUSE)
            self.p_real = self.define_forward(self.from_dataset, vreuse=tf.AUTO_REUSE)

            alpha = tf.random_uniform(
                shape = (cf.MINIBATCHSIZE,1, 1, 1),
                dtype = tf.float32,
                minval = 0.0,
                maxval = 1.0,
                name = 'C_alpha')
            
            mul1 = alpha * self.from_generator
            print('Shape of mul1:', mul1.shape)

            mul2 = (1.0 - alpha) * tf.reshape(self.from_dataset, shape=(-1, cf.PIXELSIZE, cf.PIXELSIZE, 1))

            interpolates = mul1 + mul2

            #concat_fake = tf.concat([self.from_generator, alpha], axis=4)
            #print('Shape of concat_fake:', concat_fake.shape)
            #concat_real = tf.concat([self.from_dataset, 1.0-alpha], axis=3)
                
            #interpolates = tf.reduce_prod(concat_fake, axis=3, keepdims=True) + tf.reduce_prod(concat_real, axis=3, keepdims=True)

            #interpolates = tf.matmul(alpha, self.from_dataset, transpose_a = True) + tf.matmul(1.0 - alpha, self.from_generator, transpose_a = True) 

            p_interpolates = self.define_forward(interpolates, vreuse = tf.AUTO_REUSE)
            gradients = tf.gradients(p_interpolates, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.0)**2)

            self.loss = tf.reduce_mean(self.p_fake) - tf.reduce_mean(self.p_real) + cf.LAMBDA * gradient_penalty

            tf.summary.scalar(name = 'Discriminator loss', tensor = self.loss)
            C_vars = [x for x in tf.trainable_variables() if 'C_' in x.name]

            # WGAN-GPではclip opは行わない
            #self.clip_op = [p.assign(tf.clip_by_value(p, -cf.CLIP_THRESHOLD,
            #                                          cf.CLIP_THRESHOLD))
            #                for p in C_vars]
                    
            
            #self.optimizer = tf.train.RMSPropOptimizer(learning_rate = cf.C_LEARNING_RATE,
            #                                           name = 'C_optimizer')
            self.optimizer = tf.train.AdamOptimizer(learning_rate = cf.C_LEARNING_RATE,
                                                    beta1=cf.BETA_1,
                                                    name = 'C_optimizer')

            self.train_op = self.optimizer.minimize(self.loss,
                                                    global_step=self.global_step_tensor,
                                                    var_list=C_vars,
                                                    name='C_train_op')

    def set_input_from_generator(self, generator):
        with tf.variable_scope('C_network'):
            self.from_generator = generator.output
        return

    def create_minibatch(self, session):
        '''データセットからミニバッチを作成する'''
        #minibatch_tf = self.from_dataset#f.obtain_minibatch()
        minibatch_np = session.run(self.from_dataset)
        return minibatch_np
