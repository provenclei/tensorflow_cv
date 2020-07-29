# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p78_GAN_mnist.py
@Description    :  
@CreateTime     :  2020/7/29 15:19
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import p74_framework as myf
import numpy as np


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.sample_path = 'MNIST_data'
        self.lr = 2e-5
        self.batch_size = 100
        self.keep_prob = 0.7

        self.vec_size = 4
        self.base_filters = 16
        self._ds = None

    @property
    def ds(self):
        if self._ds is None:
            self._ds = MyDS(read_data_sets(self.sample_path).train, self.vec_size)
        return self._ds

    def get_sub_tensors(self, gpu_index):
        return MySubTensors(self)

    def get_name(self):
        return 'p78'

    def random_seed(self):
        np.random.seed(9999222)

    def get_app(self):
        return MyAPP(self)

    def get_ds_train(self):
        return self.ds

    def get_ds_test(self):
        return self.ds

    def get_tensors(self):
        return MyTensors(self)


class MyDS:
    def __init__(self, ds, vec_size):
        self.num_examples = ds.num_examples
        self.ds = ds
        self.vec_size = vec_size

    def next_batch(self, batch_size):
        xs, _ = self.ds.next_batch(batch_size)
        vs = np.random.normal(size=[batch_size, self.vec_size])
        return xs, vs


class MySubTensors:
    def __init__(self, cfg: MyConfig):
        self.cfg = cfg
        x = tf.placeholder(tf.float64, [None, 784], 'x')
        v = tf.placeholder(tf.float64, [None, cfg.vec_size], 'v')
        self.inputs = [x, v]

        with tf.variable_scope('gene'):
            x2 = self.gen(v)  # [-1, 28, 28, 1]

        with tf.variable_scope('disc') as scope:
            x2_v = self.disc(x2)   # 假样本为真的概率 [-1, 1]
            x = tf.reshape(x, [-1, 28, 28, 1])
            scope.reuse_variables()
            x_v = self.disc(x)   # 真样本为真的概率 [-1, 1]

        loss1 = -tf.reduce_mean(tf.log(x_v))
        loss2 = -tf.reduce_mean(tf.log(1 - x2_v))
        loss3 = -tf.reduce_mean(tf.log(x2_v))
        self.losses = [loss1, loss2, loss3]

    def disc(self, x):
        '''

        :param x: [-1, 28, 28, 1]
        :return:
        '''
        filters = self.cfg.base_filters

        x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv1')
        for i in range(2):
            filters *= 2
            x = tf.layers.conv2d(x, filters, 3, 2, 'same',
                                 activation=tf.nn.relu, name='conv2_%d' % i)
        x = tf.layers.flatten(x)
        x = tf.nn.dropout(x, self.cfg.keep_prob)
        x = tf.layers.dense(x, 1, name='dense')  # [-1, 1]
        return x

    def gen(self, v):
        '''

        :param v: [-1, 4]
        :return:
        '''
        filters = self.cfg.base_filters * 4
        v = tf.layers.dense(v, 7*7*filters, activation=tf.nn.relu, name='dense')
        v = tf.reshape(v, [-1, 7, 7, filters])

        for i in range(2):
            filters //= 2
            v = tf.layers.conv2d_transpose(v, filters, 3, 2, 'same',
                                           activation=tf.nn.relu, name='deconv1_%d' % i)
        # [-1,28, 28, filters]
        v = tf.layers.conv2d_transpose(v, 1, 3, 1, 'same', activation=tf.nn.relu, name='deconv2')
        return v  # [-1, 28, 28, 1]


class MyAPP(myf.App):
    def before_epoch(self, epoch):
        self.config.random_seed()


class MyTensors(myf.Tensors):

    def compute_grads(self, opt):
        '''
        只对需要更新的变量进更新
        :param opt:
        :return:
        '''
        vars = tf.trainable_variables()
        vars_disc = [var for var in vars if 'disc' in var.name]
        vars_gene = [var for var in vars if 'gene' in var.name]
        vars = [vars_disc, vars_disc, vars_gene]

        # 将变量和loss对应起来
        grads = [[opt.compute_gradients(loss, vs) for vs, loss in zip(vars, ts.losses)] for ts in self.sub_ts]  # [gpus, losses]
        return [self.get_grads_mean(grads, i) for i in range(len(grads[0]))]


def main():
    MyConfig().from_cmd()


if __name__ == '__main__':
    main()