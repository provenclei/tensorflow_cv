# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p39_VAE_mnist.py
@Description    :  
@CreateTime     :  2020/6/23 10:15
------------------------------------
@ModifyTime     :  手写数字生成

"""
import p39_framework as myf
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
# from .p39_framework import tf
import numpy as np


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.sample_path = '../deeplearning_tensorflow_p/MNIST_data'
        self.vector_size = 4
        # 惯性系数
        self.momentum = 0.99

    def get_name(self):
        return 'p40'

    def get_tensors(self):
        return MyTensors(self)


class MyTensors:
    def __init__(self, config: MyConfig):
        self.config = config
        with tf.device('/gpu:0'):
            # 分配第0个gpu，而不是操作系统的第 0 块gpu
            x = tf.placeholder(tf.float32, [None, 784], 'x')
            lr = tf.placeholder(tf.float32, [], 'lr')
            self.input = [x, lr]

            x = tf.reshape(x, [-1, 28, 28, 1])
            vec = self.encode(x, config.vector_size)  # [-1, 4]
            self.process_normal(vec)  # 注意次序！！！
            y = self.decode(vec)  # [-1, 28, 28, 1]

            loss = tf.reduce_mean(tf.square(y - x))
            opt = tf.train.AdamOptimizer(lr)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                # loss,assign 在 train_op 之上定义
                # train_op 依赖 loss,assign
                self.train_op = opt.minimize(loss)
            self.summary = tf.summary.scalar('loss', tf.sqrt(loss))

    def process_normal(self, vec):
        '''
        计算平均数(动量法)
        并在计算图中保存变量（assign）
        :param vec: [-1, vec_size]
        :return:
        '''
        mean = tf.reduce_mean(vec, axis=0)  # [vec_size]
        # mean square difference
        msd = tf.reduce_mean(tf.square(vec), axis=0)

        vector_size = vec.shape[1].value
        self.final_mean = tf.get_variable('mean', [vector_size], tf.float32, tf.initializers.zeros, trainable=False)
        self.final_msd = tf.get_variable('msd', [vector_size], tf.float32, tf.initializers.zeros, trainable=False)

        mom = self.config.momentum
        # final_mean = final_mean * mom + mean * (1 - mom)   # 错误做法:变量没有更新
        assign = tf.assign(self.final_mean, self.final_mean * mom + mean * (1 - mom))
        # 建立 assign 与 train_op 的控制依赖，正向传播是走实线和虚线，反向传播不去求控制依赖的值
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign)

        assign = tf.assign(self.final_msd, self.final_msd*mom + msd*(1-mom))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign)

    def encode(self, x, vec_size):
        '''
        encode the x to vector which size is vec_size
        :param x: input tensor, shape is [-1, 28, 28, 1]
        :param vec_size:
        :return: the semantics vectors which shape is [-1, vec_size]
        '''
        filters = 16
        x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv1')  # [-1, 28, 28, 16]
        for i in range(2):
            filters *= 2
            # [-1, 28, 28, 32]  [-1, 14, 14, 64]
            x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv2_%d' % i)
            # 池化操作不产生可训练参数, 不需要训练参数
            # [-1, 14, 14, 32]   [-1, 7, 7, 64]
            x = tf.layers.max_pooling2d(x, 2, 2, 'valid')
        # x: [-1, 7, 7, 64]
        # 卷积或者使用全连接
        # [-1, 1, 1, vec_size]
        x = tf.layers.conv2d(x, vec_size, 7, 1, 'valid', name='conv3')
        return tf.reshape(x, [-1, vec_size])

    def decode(self, vec):
        '''
        使用反卷积(上采样)，反卷积只能恢复尺寸，不能恢复数值
        the semantics vector
        :param vec: [-1, vec_size]
        :return: [-1, 28, 28, 1]
        '''
        y = tf.layers.dense(vec, 7*7*64, activation=tf.nn.relu, name='dens_1')  # [-1 ,4] -> [-1, 7*7*64]
        # [-1, 7*7*64] -> [-1, 7, 7, 64]
        y = tf.reshape(y, [-1, 7, 7, 64])
        filters = 64
        for i in range(2):
            filters //= 2
            # 两次反卷积 ：[-1, 14， 14, 32]  [-1, 28, 28, 16]
            y = tf.layers.conv2d_transpose(y, filters, 3, 2, 'same', activation=tf.nn.relu, name='deconv1_%d'%i)
        # [-1, 28, 28, 16]
        y = tf.layers.conv2d_transpose(y, 1, 3, 1, 'same', name='deconv2') # [-1, 28, 28, 1]
        return y


class MyDS:
    def __init__(self, ds, config):
        self.ds = ds
        self.lr = config.lr
        self.num_examples = ds.num_examples

    def next_batch(self, batch_size):
        xs, ys = self.ds.next_batch(batch_size)
        return xs, self.lr


class App:
    def __init__(self):
        pass


if __name__ == '__main__':
    cfg = MyConfig()
    cfg.from_cmd()
    print('_'*20)
    print(cfg)

    dss = read_data_sets(cfg.sample_path)
    app = myf.App(cfg)
    with app:
        app.train(MyDS(dss.train, cfg), MyDS(dss.validation, cfg))
        mean = app.session.run(app.ts.final_mean)
        print(mean)
        msd = app.session.run(app.ts.final_msd)  # 二阶原点矩
        print(np.sqrt(msd - mean**2))
