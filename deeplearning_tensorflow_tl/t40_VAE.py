# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t40_VAE.py
@Description    :  
@CreateTime     :  2020/6/23 18:22
------------------------------------
@ModifyTime     :  
"""
from TF_turial.deeplearning_tensorflow_tl import t39_framework_2 as myf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import tensorflow as tf


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.sample_path = '../deeplearning_tensorflow_p/MNIST_data'
        self.vector_size = 4
        self.momentum = 0.99

    def get_name(self):
        return 'p40'

    def get_tensors(self):
        return MyTensors(self)


class MyTensors:
    def __init__(self, config: MyConfig):
        self.config = config
        with tf.device('/gpu:0'):
            x = tf.placeholder(tf.float32, [None, 784], 'x')
            lr = tf.placeholder(tf.float32, [], 'lr')
            self.input = [x, lr]

            x = tf.reshape(x, [-1, 28, 28, 1])
            # [-1, 28, 28, 1] -> [-1, 4]
            vec = self.encode(x, config.vector_size)
            # 计算均值
            self.normal_process(vec)
            # [-1, 4] -> [-1, 28, 28, 1]
            y = self.decode(vec)

            loss = tf.reduce_mean(tf.square(y - x))
            opt = tf.train.AdamOptimizer(lr)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                # 每次计算loss之后更新变量
                self.train_op = opt.minimize(loss)
            self.summary = tf.summary.scalar('loss', tf.sqrt(loss))

    def encode(self, vec, vec_size):
        '''
        [-1, 28, 28, 1] -> [-1, 4]
        :param vec:
        :param vec_size:
        :return:
        '''
        filters = 16
        x = tf.layers.conv2d(vec, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv1')  # [-1, 28, 28, 16]
        for i in range(2):
            filters *= 2
            # [-1, 28, 28, 32]  [-1, 14, 14, 64]
            x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv2_%d' % i)
            # [-1, 14, 14, 32]  [-1, 7, 7, 64]
            x = tf.layers.max_pooling2d(x, 2, 2, 'valid')
        x = tf.layers.conv2d(x, vec_size, 7, 1, 'valid')
        return tf.reshape(x, [-1, vec_size])

    def decode(self, vec):
        '''
        [-1, 4] -> [-1, 28, 28, 1]
        :param vec:
        :return:
        '''
        # [-1 ,4] -> [-1, 7*7*64]
        y = tf.layers.dense(vec, 7*7*64, activation=tf.nn.relu, name='dense_1')
        y = tf.reshape(y, [-1, 7, 7, 64])
        filters = 64
        for i in range(2):
            filters //= 2
            # [-1, 14, 14, 32]  [-1, 28, 28, 16]
            y = tf.layers.conv2d_transpose(y, filters, 3, 2, 'same', activation=tf.nn.relu, name='trans_conv1_%d' % i)
        # [-1, 28, 28, 16] -> [-1, 28, 28, 1]
        y = tf.layers.conv2d_transpose(y, 1, 3, 1, 'same', name='trans_conv2')
        return y

    def normal_process(self, vec):
        '''
        使用动量法 计算均值
        :return:
        '''
        mean = tf.reduce_mean(vec, axis=0)
        vec_size = vec.shape[1]
        mean_var = tf.get_variable('mean', [vec_size], tf.float32, tf.initializers.zeros, trainable=False)
        # 动量法
        mom = self.config.momentum
        assign = tf.assign(mean_var, mean_var * mom + mean * (1 - mom))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign)


class MyDS:
    def __init__(self, ds, config):
        self.ds = ds
        self.lr = config.lr
        self.num_examples = ds.num_examples

    def next_batch(self, batch_size):
        xs, _ = self.ds.next_batch(batch_size)
        return xs, self.lr


def main():
    cfg = MyConfig()
    cfg.cmd_parmerter()
    print('-' * 100)
    print(cfg)

    dss = read_data_sets(cfg.sample_path)
    app = myf.App(cfg)
    with app:
        app.train(MyDS(dss.train, cfg), MyDS(dss.validation, cfg))


if __name__ == '__main__':
    main()