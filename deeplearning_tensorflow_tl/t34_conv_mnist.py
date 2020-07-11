# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t33_conv_cos.py
@Description    :  
@CreateTime     :  2020/6/18 19:23
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Config:
    def __init__(self):
        self.save_path = './models/p33/1/conv_mnist'
        self.data_path = '../deeplearning_tensorflow_p/MNIST_data'
        self.lr = 0.01
        self.epoches = 1000
        self.batch_size = 200


class Sample:
    def __init__(self, config: Config):
        self.data = read_data_sets(config.data_path)


class Tensors:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.int32, [None], name='y')


        x = tf.reshape(self.x, [-1, 28, 28, 1])
        # [-1, 28, 28, 16]
        x = tf.layers.conv2d(x, 16, 3, 1, 'same', activation=tf.nn.leaky_relu)
        # [-1, 14, 14, 32]
        x = tf.layers.conv2d(x, 32, 3, 2, activation=tf.nn.leaky_relu)
        # [-1, 7, 7, 64]
        x = tf.layers.conv2d(x, 64, 3, 2, activation=tf.nn.leaky_relu)

        # [-1, 7*7*64]
        x = tf.layers.flatten(x)
        # [-1, 10]
        logits = tf.layers.dense(x, 10)
        y = tf.one_hot(self.y, 10)   # [-1, 10]
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

        self.lr = tf.placeholder(tf.float32, (), name='lr')
        opt = tf.train.AdamOptimizer(self.lr)
        self.opt_train = opt.minimize(self.loss)

        # [-1]
        self.y_predict = tf.argmax(logits, axis=1, output_type=tf.int32)
        self.precise = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_predict), tf.float32))


class MNISTApp:
    def __init__(self, config: Config):
        self.config = config
        self.ts = Tensors()
        self.session = tf.Session()
        sample = Sample(config)
        self.data = sample.data
        self.saver = tf.train.Saver()

        try:
            self.saver.restore(self.session, config.save_path)
            print('模型读取成功')
        except:
            self.session.run(tf.global_variables_initializer())
            print('模型初始化成功')

    def train(self):
        batches = self.data.train.num_examples // self.config.batch_size
        for epoch in range(self.config.epoches):
            for batch in range(batches):
                xs, ys = self.data.train.next_batch(self.config.batch_size)
                _, loss, precise = self.session.run([self.ts.opt_train, self.ts.loss, self.ts.precise], {
                    self.ts.x: xs,
                    self.ts.y: ys,
                    self.ts.lr: self.config.lr
                })
                print('%d/%03d. loss = %.6f,  precise = %.6f' % (epoch, batch, loss, precise))
            self.save()
        return self.ts.x, self.ts.y_predict

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('Save model into', self.config.save_path)

    def predict(self):
        precise_total = 0
        batches = self.data.test.num_examples // self.config.batch_size
        for batch in range(batches):
            xs, ys = self.data.test.next_batch(self.config.batch_size)
            precise = self.session.run(self.ts.precise, {
                self.ts.x: xs,
                self.ts.y: ys
            })
            precise_total += precise
        print('precise_total:', precise_total/self.ts.batch_size)
        return self.ts.x, self.ts.y_predict

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    config = Config()
    app = MNISTApp(config)
    with app:
        xs_train, ys_train = app.train()
        xs_test, ys_test = app.predict()

        ys_train = np.transpose(ys_train)
        plt.plot(xs_train, ys_train)
        plt.plot(xs_test, ys_test)
        plt.show()


if __name__ == '__main__':
    main()