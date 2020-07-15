# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p62_inception_v3.py
@Description    :  
@CreateTime     :  2020/7/15 09:36
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf


_name_id = 1


def inception(x, filters=64, models=5, name=None):
    # x: [-1, h, w, 3]
    if name is None:
        global _name_id
        name = 'inception_v3_%d' % _name_id
        _name_id += 1
    with tf.variable_scope(name):
        for i in range(models):
            x = inception_module(x, filters, '%s_%d' % (name, i))
            filters *= 2
            print(x.shape)
        return x


def inception_module(x, filters, name):
    with tf.variable_scope(name):
        # [-1, h/2, w/2, 64]
        branch1 = tf.layers.conv2d(x, filters, 1, 2, 'same', name='branch1')

        branch2 = tf.layers.max_pooling2d(x, x.shape[-1].value, 1, 'same', name='branch2_pool')
        branch2 = tf.layers.conv2d(branch2, filters, 1, 2, 'same', name='branch2_conv')

        branch3 = tf.layers.conv2d(x, x.shape[-1].value, 1, 1, 'same', name='branch3_conv1')
        branch3 = tf.layers.conv2d(branch3, filters, 3, 2, 'same', name='branch3_conv2')

        branch4 = tf.layers.conv2d(x, x.shape[-1].value, 1, 1, 'same', name='branch4_conv1')
        branch4 = tf.layers.conv2d(branch4, filters, 3, 1, 'same', name='branch4_conv2')
        branch4 = tf.layers.conv2d(branch4, filters, 3, 2, 'same', name='branch4_conv3')
        return tf.nn.relu(branch1 + branch2 + branch3 + branch4)


def main():
    x = tf.random_normal([20, 224, 224, 3])
    print('input:', x.shape)
    y = inception(x)
    print('output:', y.shape)


if __name__ == '__main__':
    main()