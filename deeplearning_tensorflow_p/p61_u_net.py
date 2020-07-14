# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p61_u_net.py
@Description    :  
@CreateTime     :  2020/7/14 10:51
------------------------------------
@ModifyTime     :  UNet 实现
"""
import tensorflow as tf

_name_id = 1


def unet(x, convs=4, base_filters=64, name=None):
    """
    UNet 实现
    :param x: x: [n, h, w, c]
    :param conv:
    :param name:
    :return:
    """
    if name is None:
        global _name_id
        name = 'unet_%d' % _name_id
        _name_id += 1

    with tf.variable_scope(name):
        semantics, stack = encode(x, convs, base_filters)
        return decode(semantics, stack)


def encode(x, convs, filters):
    x = tf.layers.conv2d(x, filters, 3, 1, 'same', name='conv1', activation=tf.nn.relu)
    stack = []
    for i in range(convs):
        filters *= 2
        x = tf.layers.conv2d(x, filters, 3, 2, 'same', name='conv2_%d' % i, activation=tf.nn.relu)
        stack.append(x)
    return x, stack


def decode(semantics, stack):
    stack = reversed(stack)
    filters = semantics.shape[-1].value
    y = semantics
    for i, x in enumerate(stack):
        y += x
        filters //= 2
        y = tf.layers.conv2d_transpose(x, filters, 3, 2, 'same', name='deconv1_%d' % i, activation=tf.nn.relu)
    y = tf.layers.conv2d_transpose(y, 3, 3, 1, 'same', name='deconv2')
    return y


def main():
    x = tf.random_normal([20, 128, 128, 3])
    y = unet(x)
    print(y.shape)


if __name__ == '__main__':
    main()