# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p31_mnist.py
@Description    :  
@CreateTime     :  2020/6/15 09:14
------------------------------------
@ModifyTime     :  
"""
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import cv2
import numpy as np
import tensorflow as tf


class Tensors:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784])
        x = tf.layers.dense(self.x, 2000, tf.nn.relu)  # [-1, 2000]
        x = tf.layers.dense(x, 10)  # [-1, 10]
        y_predict = tf.nn.softmax(x)
        y2 = tf.maximum(y_predict, 1e-6)   # [-1, 10]

        self.y = tf.placeholder(tf.int32, [None])  # [-1]
        y = tf.one_hot(self.y, 10)   # [-1, 10]

        self.loss = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y2), axis=1))
        self.lr = tf.placeholder(tf.float32, ())
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = opt.minimize(self.loss)


class MNISTApp:
    def __init__(self):
        self.ts = Tensors()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def train(self, ds, lr=0.001, epoches=2000, batch_size=200):
        for epoche in epoches:
            batches = ds.train.num_examples // batch_size
            for batch in range(batches):
                xs, ys = ds.train.next_batch(batch_size)
                _, loss = \
                self.session.run([self.ts.train_op, self.ts.loss], {self.ts.x: xs, self.ts.y: ys, self.ts.lr: lr})
                print('%d, %d, loss:%.5f' % (epoche, batch, loss))

    def predict(self):
        pass


def main():
    path = './MNIST_data'
    ds = read_data_sets(path)
    # 55000
    # 10000
    # 5000
    print(ds.train.num_examples)
    print(ds.test.num_examples)
    print(ds.validation.num_examples)

    # imgs, labels = ds.train.next_batch(1)  # [-1, 784]
    # imgs = np.reshape(imgs, [-1, 28, 28])
    # print(imgs.shape, imgs[0])
    # print(labels.shape, labels)
    #
    # imgs2, labels2 = ds.train.next_batch(10)
    # imgs2 = np.reshape(imgs2, [-1, 28])
    #
    # # 两行十列
    # imgs3, labels3 = ds.train.next_batch(20)
    # imgs3 = np.reshape(imgs3, [-1, 28, 28])
    # imgs3 = np.transpose(imgs3, [1, 0, 2])  # [28, -1, 28]
    # imgs3 = np.reshape(imgs3, [28, -1, 28*2])
    # imgs3 = np.transpose(imgs3, [1, 0, 2])  # [-1, 28, 56]
    # imgs3 = np.reshape(imgs3, [-1, 28*2])
    # print(imgs3, labels3)
    #
    # # 十列两行
    # imgs4, labels4 = ds.train.next_batch(20)  # [-1， 784]
    # imgs4 = np.reshape(imgs4, [-1, 28, 28])
    # imgs4 = np.transpose(imgs4, [1, 0, 2])  # [28, -1, 28]
    # imgs4 = np.reshape(imgs4, [28, -1, 28 * 10])
    # imgs4 = np.transpose(imgs4, [1, 0, 2])  # [-1, 28, 280]
    # imgs4 = np.reshape(imgs4, [-1, 28 * 10])
    # print(imgs4, labels4)
    #
    # cv2.imshow('ABC', imgs4)
    # cv2.waitKey()
    #
    # cv2.imshow('ABC', imgs3)
    # cv2.waitKey()
    #
    # cv2.imshow('ABC', imgs2)
    # cv2.waitKey()
    #
    # cv2.imshow('ABC', imgs[0])
    # cv2.waitKey()
    app = MNISTApp()
    app.train(ds)
    app.predict()


if __name__ == '__main__':
    main()