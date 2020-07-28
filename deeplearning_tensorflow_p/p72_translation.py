# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p72_translation.py
@Description    :  
@CreateTime     :  2020/7/27 14:01
------------------------------------
@ModifyTime     :  多对一模型：翻译模型(张量部分)
                双层 LSTM
"""
import tensorflow as tf


class MyConfig:
    def __init__(self):
        self.num_steps1 = 50
        self.num_steps2 = 60
        self.num_units = 200
        self.words1 = 6000
        self.words2 = 7000
        self.keep_prob = 0.6


class MySubTensors:
    def __init__(self, cfg: MyConfig):
        self.x = tf.placeholder(tf.int64, [None, cfg.num_steps1], 'x')   # 输入 [-1, num_steps1]
        self.y = tf.placeholder(tf.int64, [None, cfg.num_steps2], 'y')   # 标签

        # 可以换成已训练好的词向量字典
        dict1 = tf.get_variable('dict1', [cfg.words1, cfg.num_units], tf.float64)  # 单词向量字典 [6000, 200]
        # 不可训练操作，不产生变量
        #  x: [-1, steps1, units]
        x = tf.nn.embedding_lookup(dict1, self.x)  # use word2vec instead of one-hot and dense

        cell1 = [tf.nn.rnn_cell.LSTMCell(cfg.num_units, name='cell1_%d' % i) for i in range(2)]   # [cell11, cell12]
        cell1 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(cell1), cfg.keep_prob)

        # encode
        batch_size = tf.shape(self.x)[0]
        state = cell1.zero_state(batch_size, x.dtype)
        for i in range(cfg.num_steps1):
            #  x: [-1, steps1, units]
            _, state = cell1(x[:, i, :], state)  # [-1, units]

        # decode
        cell2 = [tf.nn.rnn_cell.LSTMCell(cfg.num_units, name='cell2_%d' % i) for i in range(2)]
        cell2 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(cell2), cfg.keep_prob)
        # 没有输入，即为0
        zero = tf.zeros([batch_size, cfg.num_units], x.dtype)
        y_predict = []
        loss = []
        y = tf.one_hot(self.y, cfg.words2)  # y: [-1, step2] -> [-1, steps2, words2]
        for i in range(cfg.num_steps2):
            yi, state = cell2(zero, state)  # [-1, units]

            logits = tf.layers.dense(yi, cfg.words2, name='dense1')  # [-1, words2]
            # 命名空间共享
            tf.get_variable_scope().reuse_variables()

            yi_predict = tf.argmax(logits, axis=1)  # [-1]
            # y: [steps2, -1]
            y_predict.append(yi_predict)

            # y[:, i, :] -> [-1, word2]  logits -> [-1, words2]
            loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y[:, i, :], logits=logits)
            loss.append(loss_i)

        self.loss = tf.reduce_mean(loss)
        self.y_predict = tf.transpose(y_predict)  # [-1, steps2]
        self.losses = [self.loss]


if __name__ == '__main__':
    MySubTensors(MyConfig())

    for var in tf.trainable_variables():
        print(var.name, var.shape)
