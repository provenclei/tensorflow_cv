# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p73_self_attention.py
@Description    :  
@CreateTime     :  2020/7/27 17:44
------------------------------------
@ModifyTime     :  self-attention 实现
"""
import tensorflow as tf


name_id = 1


def self_attention(inputs, num_steps2: int, name=None):
    # inputs: [steps1, -1, units]
    # return: [steps2, -1, units]
    if name is None:
        global name_id
        name = 'self_attention_%d' % name_id
        name_id += 1

    with tf.variable_scope(name):
        inputs = tf.transpose(inputs, [1, 0, 2])  # [-1, steps1, units]
        # 不需要偏置和激活函数
        attention = tf.layers.dense(inputs, num_steps2, name='dense1', use_bias=False)  # [-1, steps1, steps2]
        attention = tf.nn.softmax(attention, axis=1)  # [-1, steps1, steps2]

        inputs = tf.transpose(inputs, [0, 2, 1])  # [-1, units, steps1]
        inputs = tf.matmul(inputs, attention)  # [-1, units, steps2]
        return tf.transpose(inputs, [2, 0, 1])  # [steps2, -1, units]


if __name__ == '__main__':
    inputs = tf.random_normal([50, 123, 200])
    outputs = self_attention(inputs, 60)
    print(outputs.shape)  # [60, 123, 200)
