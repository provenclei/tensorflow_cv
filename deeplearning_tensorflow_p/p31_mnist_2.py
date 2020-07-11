# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p31_mnist_2.py
@Description    :  
@CreateTime     :  2020/6/15 13:02
------------------------------------
@ModifyTime     :  
"""
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import numpy as np
import cv2
import tensorflow as tf


def main():
    path = './MNIST_data'
    data = read_data_sets(path)
    print(data.train.num_examples)
    print(data.test.num_examples)
    print(data.validation.num_examples)

    # 显示一张图片
    # images1, labels1 = data.train.next_batch(10)
    # images1 = np.reshape(images1, [-1, 28, 28])
    # print(images1.shape, labels1.shape)
    # cv2.imshow('a1', images1[0])

    # 一列十张
    # images2, labels2 = data.train.next_batch(10)
    # images2 = np.reshape(images2, [-1, 28])
    # print(images2.shape, labels2.shape)
    # cv2.imshow('a2', images2)

    # 两行十列  [-1, 28*2]
    # images3, labels3 = data.train.next_batch(20)
    # images3 = np.reshape(images3, [-1, 28, 28])
    # images3 = tf.transpose(images3, [1, 0, 2])  # [28, -1, 28]
    # images3 = np.reshape(images3, [28, -1, 28*2])
    # images3 = tf.transpose(images3, [1, 0, 2])  # [-1, 28, 28*2]
    # images3 = np.reshape(images3, [-1, 28*2])
    # print(images3.shape, labels3.shape, labels3)
    # cv2.imshow('a3', images3)

    # opencv需要的参数为 numpy 类型
    # image3, labels2 = data.train.next_batch(20)
    # image3 = tf.reshape(image3, [-1, 28, 28])
    # image3 = tf.transpose(image3, [1, 0, 2])
    # image3 = tf.reshape(image3, [28, -1, 28*2])
    # image3 = tf.transpose(image3, [1, 0, 2])
    # image3 = tf.reshape(image3, [-1, 28*2])
    # session = tf.Session()
    # image3_np = image3.eval(session=session)
    # print(type(image3_np))
    # cv2.imshow('a3', image3_np)

    # 十行两列
    image4, labels4 = data.train.next_batch(20)
    image4 = np.reshape(image4, [-1, 28, 28])
    image4 = np.transpose(image4, [1, 0, 2])
    image4 = np.reshape(image4, [28, -1, 28 * 10])
    image4 = np.transpose(image4, [1, 0, 2])
    image4 = np.reshape(image4, [-1, 28 * 10])
    cv2.imshow('a3', image4)
    cv2.waitKey()


if __name__ == '__main__':
    main()