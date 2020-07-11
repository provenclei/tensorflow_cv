# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t31_mnist.py
@Description    :  
@CreateTime     :  2020/6/15 15:36
------------------------------------
@ModifyTime     :  
"""
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import tensorflow as tf
import cv2
import numpy as np


def main():
    path = '../artifitial_intelligent_p.MNIST_data'
    data = read_data_sets(path)
    print(data.train.num_examples, data.test.num_examples, data.validation.num_examples)

    # 读取一张图片
    # img_1, label_1 = data.train.next_batch(10)  # []
    # img_1 = np.reshape(img_1, [-1, 28, 28])
    # cv2.imshow('img1', img_1[0])

    # 读取十张图片（十行一列）
    # img_2, label_2 = data.train.next_batch(10)
    # img_2 = np.reshape(img_2, [-1, 28])
    # cv2.imshow('img2', img_2)

    # 十行两列
    # img_3, label_3 = data.train.next_batch(20)
    # img_3 = np.reshape(img_3, [-1, 28, 28])
    # img_3 = np.transpose(img_3, [1, 0, 2])  # [28, -1, 28]
    # img_3 = np.reshape(img_3, [28, -1, 28*10])
    # img_3 = np.transpose(img_3, [1, 0, 2])  # [-1, 28, 28*10]
    # img_3 = np.reshape(img_3, [-1, 28*10])  # [-1, 28*10]
    # cv2.imshow('img3', img_3)

    # 两行十列
    img_4, label_4 = data.train.next_batch(20)
    img_4 = np.reshape(img_4, [-1, 28, 28])
    img_4 = np.transpose(img_4, [1, 0, 2])  # [28, -1, 28]
    img_4 = np.reshape(img_4, [28, -1, 28 * 2])
    img_4 = np.transpose(img_4, [1, 0, 2])  # [-1, 28, 28*2]
    img_4 = np.reshape(img_4, [-1, 28 * 2])
    cv2.imshow('img4', img_4)

    cv2.waitKey()


if __name__ == '__main__':
    main()