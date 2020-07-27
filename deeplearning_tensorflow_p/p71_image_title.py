# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p71_image_title.py
@Description    :  
@CreateTime     :  2020/7/26 22:31
------------------------------------
@ModifyTime     :  一对多模型：图片生成标题的框架
"""
import tensorflow as tf
from p62_inception_v3 import inception


class MySubTensors:
    def __init__(self):
        self.x = tf.placeholder(tf.float64, [None, 224, 224, 3], 'x')
        self.y = tf.placeholder(tf.int64, [None, 50], 'y')

        x = inception(self.x, name='inception')
        x = tf.layers.flatten(x)
        x = tf.nn.dropout(x, 0.6)
        x = tf.layers.dense(x, 200, name='dense1')  # [-1, 200]

        y = tf.one_hot(self.y, 4340)  # [-1, 50, 4340]

        cell1 = tf.nn.rnn_cell.LSTMCell(200, name='cell1')
        cell2 = tf.nn.rnn_cell.LSTMCell(200, name='cell2')
        cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

        state = cell.zero_state(tf.shape(x)[0], x.dtype)
        losses = []
        for i in range(50):
            yi_predict, state = cell(x, state)
            yi_predict = tf.layers.dense(yi_predict, 4340, name='dense2')  # [-1, 4340]
            tf.get_variable_scope().reuse_variables()

            lossi = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y[:, i, :], logits=yi_predict)
            losses.append(lossi)

        loss = tf.reduce_mean(losses)


if __name__ == '__main__':
    ts = MySubTensors()
