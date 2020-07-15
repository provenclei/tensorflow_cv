# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t62_inception.py
@Description    :  
@CreateTime     :  2020/7/14 17:30
------------------------------------
@ModifyTime     :  Inception v3
"""
import tensorflow as tf
import numpy as np


_name_id = 1


class InceptionNet:
    def __init__(self, filters=64, convs=5):
        self.filters = 64
        self.convs = 5

    def inception(self, x, filters=64, convs=5, name=None):
        if name is None:
            global _name_id
            name = 'inception_%d' % _name_id
            _name_id += 1

        with tf.variable_scope(name):
            for i in range(convs):
                # [-1, 228, 228, 3] -> [-1, 112, 112, 64]
                x = self.incep_modual(x, filters, '%s_%d' % (name, i))
                filters *= 2
                print(x.shape)
            return x

    def incep_modual(self, x, filters, name):
        with tf.variable_scope(name):
            part1 = tf.layers.conv2d(x, filters, 1, 2, 'same', name='p1_conv1')

            part2 = tf.layers.max_pooling2d(x, x.shape[-1].value, 1, 'same', name='pooling')
            part2 = tf.layers.conv2d(part2, filters, 1, 2, 'same', name='p2_conv1')

            part3 = tf.layers.conv2d(x, x.shape[-1].value, 1, 1, 'same', name='p3_conv1')
            part3 = tf.layers.conv2d(part3, filters, 3, 2, 'same', name='p3_conv2')

            part4 = tf.layers.conv2d(x, x.shape[-1].value, 1, 1, 'same', name='p4_conv1')
            part4 = tf.layers.conv2d(part4, filters, 3, 1, 'same', name='p4_conv2')
            part4 = tf.layers.conv2d(part4, filters, 3, 2, 'same', name='p4_conv3')
            y = tf.nn.relu(part1 + part2 + part3 + part4)
            return y


def main():
    incep = InceptionNet()
    x = tf.random_normal([200, 224, 224, 3])
    result = incep.inception(x)


if __name__ == '__main__':
    main()