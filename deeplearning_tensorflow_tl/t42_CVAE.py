# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t41_VAE_test.py
@Description    :  
@CreateTime     :  2020/6/28 15:05
------------------------------------
@ModifyTime     :  实现条件VAE：CVAE
"""
from TF_turial.deeplearning_tensorflow_tl import t39_framework_2 as myf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import tensorflow as tf
import numpy as np
import cv2


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.sample_path = '../deeplearning_tensorflow_p/MNIST_data'
        self.vector_size = 4
        self.momentum = 0.99
        self.cols = 20
        self.img_path = './imgs/{name}/test.jpg'.format(name=self.get_name())
        self.batch_size = 200
        self.epoches = 2

    def get_name(self):
        return 't42'

    def get_tensors(self):
        return MyTensors(self)


class MyTensors:
    def __init__(self, config: MyConfig):
        self.config = config
        with tf.device('/gpu:0'):
            x = tf.placeholder(tf.float32, [None, 784], 'x')
            lr = tf.placeholder(tf.float32, [], 'lr')
            label = tf.placeholder(tf.int32, [None], 'label')

            self.input = [x, lr, label]
            x = tf.reshape(x, [-1, 28, 28, 1])
            self.vec = self.encoder(x, config.vector_size)  # [-1, 4]
            # 处理均值，标准差
            self.process_normal(self.vec)
            self.y = self.decoder(self.vec, label)  # [-1, 28, 28, 1]

            loss = tf.reduce_mean(tf.square(self.y - x))
            opt = tf.train.AdamOptimizer(lr)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = opt.minimize(loss)
            self.summary = tf.summary.scalar('loss', tf.sqrt(loss))
            self.y = tf.reshape(self.y, [-1, 28, 28])

    def process_normal(self, vec):
        mean = tf.reduce_mean(vec, axis=0)
        msd = tf.reduce_mean(tf.square(vec), axis=0)
        vec_size = vec.shape[1].value
        self.final_mean = tf.get_variable('mean', [vec_size], tf.float32, tf.initializers.zeros, trainable=False)
        self.final_msd = tf.get_variable('msd', [vec_size], tf.float32, tf.initializers.zeros, trainable=False)

        mom = self.config.momentum
        assign = tf.assign(self.final_mean, self.final_mean*mom + mean*(1-mom))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign)
        assign = tf.assign(self.final_msd, self.final_msd*mom + msd*(1-mom))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign)

    def encoder(self, x, vec_size):
        '''
        [-1, 28, 28, 1]  -> [-1, 4]
        :param
        vec:
        :return:
        '''
        filters = 16
        # [-1, 28, 28, 16]
        x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv_1')
        for i in range(2):
            filters *= 2
            x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv_2_%d'%i)
            x = tf.layers.max_pooling2d(x, 2, 2, 'valid')
        # [-1, 7, 7, 64]
        x = tf.layers.conv2d(x, vec_size, 7, 1, 'valid', name='conv_3')
        return tf.reshape(x, [-1, vec_size])

    def decoder(self, vec, label):
        '''
        [-1, 4]  ->  [-1, 28, 28, 1]
        :param vec:
        :return:
        '''
        y = tf.layers.dense(vec, 7 * 7 * 64, activation=tf.nn.relu, name='dense_1')
        label = tf.one_hot(label, 10)
        l = tf.layers.dense(label, 7 * 7 * 64, name='dense1')
        y += l
        y = tf.reshape(y, [-1, 7, 7, 64])

        filters = 64
        size = 7
        for i in range(2):
            filters //= 2
            size *= 2
            y = tf.layers.conv2d_transpose(y, filters, 3, 2, 'same', activation=tf.nn.relu, name='contrans_1_%d'%i)
            l = tf.layers.dense(label, size * size * filters, name='deconv_l_1_%d' % i)
            l = tf.reshape(l, [-1, size, size, filters])
            y += l
        # [-1, 28, 28, 16]
        y = tf.layers.conv2d_transpose(y, 1, 3, 1, 'same', name='contrans_2')
        return y


class App:
    def __init__(self):
        pass


class MyDS:
    def __init__(self, ds, config: MyConfig):
        self.lr = config.lr
        self.ds = ds
        self.num_examples = ds.num_examples

    def next_batch(self, batch_size):
        xs, labels = self.ds.next_batch(batch_size)
        return xs, self.lr, labels


def predict(app, samples, path, cols):
    mean = app.session.run(app.ts.final_mean)
    print(mean)
    msd = app.session.run(app.ts.final_msd)
    std = np.sqrt(msd - mean**2)
    print(std)

    vec = np.random.normal(mean, std, [samples, len(std)])
    label = [i % 10 for i in range(samples)]
    img = app.session.run(app.ts.y, {app.ts.vec: vec, app.ts.input[-1]: label})
    # 展示 [-1, 28, 28]
    img = np.reshape(img, [-1, cols, 28, 28])
    img = np.transpose(img, [0, 2, 1, 3])  # [-1, 28, 20, 28]
    img = np.reshape(img, [-1, cols*28])
    cv2.imwrite(path, img*255)


def main():
    cfg = MyConfig()
    print(cfg)
    cfg.cmd_parmerter()
    print("-"*30)
    print(cfg)

    app = myf.App(cfg)
    dss = read_data_sets(cfg.sample_path)
    with app:
        # app.train(MyDS(dss.train, cfg), MyDS(dss.validation, cfg))
        predict(app, cfg.batch_size, cfg.img_path, cfg.cols)


if __name__ == '__main__':
    main()