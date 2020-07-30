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


def attention(input, query, type, name=None):
    '''
    注意力机制
    :param input: [batch_size, num_steps, vec_size]
    :param query: [vec_size, 1]
    :param type: {'dot', 'scale dot-product', 'general', 'concat'}
    :param name:
    :return: [batch_size, vec_size]
    '''
    if name == None:
        global name_id
        name = 'self_attention_%d' % name_id
        name_id += 1

    with tf.variable_scope(name):
        key = tf.layers.dense(input, vec_size, name='dense1', use_bias=False)  # [batch_size, num_steps, vec_size]
        value = key  # [batch_size, num_steps, vec_size]

        # query: [vec_size, 1]
        # key: [batch_size, num_steps, vec_size]
        # value: [batch_size, num_steps, vec_size]
        if type == 'dot':
            score = tf.matmul(key, query)  # [-1, num_steps, 1]
        elif type == 'scale dot-product':
            score = tf.matmul(key, query)/tf.square(key.shape[-1])  # [-1, num_steps, 1]
        elif type == 'general':
            # 双线性插值  key * w * query
            pass
        elif type == 'concat':
            # v * tanh(concat(key, query))
            q = tf.transport(query, [1, 0])
            q = tf.reshape(q, [1, 1, q.shape[0].value])  # [1, 1, vec_size]
            c = tf.concat(key, q, axis=2)  # [batch_size, num_steps, 2 * vec_size]
            # [batch_size, num_steps, vec_size]
            c = tf.layers.dense(c, vec_size, name='dense1', activation=tf.nn.tanh, use_bias=False)

        elif type == 'add':
            # v * tanh(key + query)
            q = tf.transport(query, [1, 0])
            q = tf.reshape(q, [1, 1, q.shape[0].value])  # [1, 1, vec_size]
            c = key + q  # [batch_size, num_steps, vec_size]

        else:
            raise Exception('没有此类型')
        # [-1, num_steps, 1]
        s = tf.nn.softmax(score, axis=1)
        v = tf.transpose(value, [0, 2, 1])  # [-1, vec_size, num_steps]
        att = tf.matmul(s, v)  # [-1, vec_size, 1]
        att = tf.reshape(att, [att.shape[0].value, att.shape[1].value])  # [-1, vec_size]
        return att


if __name__ == '__main__':
    inputs = tf.random_normal([50, 123, 200])
    outputs = self_attention(inputs, 60)
    print(outputs.shape)  # [60, 123, 200)
