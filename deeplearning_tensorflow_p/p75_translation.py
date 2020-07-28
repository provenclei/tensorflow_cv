# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p72_translation.py
@Description    :  
@CreateTime     :  2020/7/27 14:01
------------------------------------
@ModifyTime     :  使用framework
                多对一模型：翻译模型(张量部分)
                双层 LSTM
"""
import tensorflow as tf
import numpy as np
import p74_framework as myf
from p73_self_attention import self_attention


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.num_steps1 = 50
        self.num_steps2 = 60
        self.num_units = 200
        self.words1 = 6000
        self.words2 = 7000
        self.keep_prob = 0.6
        self._ds = None
        self.batch_size = 100
        self.lr = 0.0001

    def get_name(self):
        return 'p75'

    def get_sub_tensors(self, gpu_index):
        return MySubTensors(self)

    def get_ds_train(self):
        return self.ds

    def get_ds_test(self):
        return self.ds

    @property
    def ds(self):
        if self._ds is None:
            self._ds = MyDS(self)
        return self._ds


class MySubTensors:
    def __init__(self, cfg: MyConfig):
        self.x = tf.placeholder(tf.int64, [None, cfg.num_steps1], 'x')
        self.y = tf.placeholder(tf.int64, [None, cfg.num_steps2], 'y')
        self.inputs = [self.x, self.y]

        dict1 = tf.get_variable('dict1', [cfg.words1, cfg.num_units], tf.float64)  # 单词向量字典
        x = tf.nn.embedding_lookup(dict1, self.x)  # [-1, steps1, units], use word2vec instead of one-hot and dense

        cell1 = [tf.nn.rnn_cell.LSTMCell(cfg.num_units, name='cell1_%d' % i) for i in range(2)]
        cell1 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(cell1), cfg.keep_prob)

        # encode
        batch_size = tf.shape(self.x)[0]
        state = cell1.zero_state(batch_size, x.dtype)
        enc_out = []
        for i in range(cfg.num_steps1):
            out, state = cell1(x[:, i, :], state)
            enc_out.append(out)
        # enc_out: [steps1, -1, units]
        dec_ins = self_attention(enc_out, cfg.num_steps2, 'self_attention')  # [steps2, -1, units]

        # decode
        cell2 = [tf.nn.rnn_cell.LSTMCell(cfg.num_units, name='cell2_%d' % i) for i in range(2)]
        cell2 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(cell2), cfg.keep_prob)
        y_predict = []
        loss = []
        y = tf.one_hot(self.y, cfg.words2)  # [-1, steps2, words2]
        for i in range(cfg.num_steps2):
            yi, state = cell2(dec_ins[i, :, :], state)  # [-1, units]

            logits = tf.layers.dense(yi, cfg.words2, name='dense1')  # [-1, words2]
            tf.get_variable_scope().reuse_variables()

            yi_predict = tf.argmax(logits, axis=1)  # [-1]
            y_predict.append(yi_predict)

            loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y[:, i, :], logits=logits)
            loss.append(loss_i)

        # y: [steps2, -1]
        self.loss = tf.reduce_mean(loss)
        self.y_predict = tf.transpose(y_predict)  # [-1, steps2]
        self.losses = [self.loss]


class MyDS:
    def __init__(self, cfg:MyConfig):
        self.config = cfg
        self.batches = 10000
        self.num_examples = self.batches * cfg.batch_size
        self.batch = 0
        self._seed()

    def _seed(self):
        np.random.seed(980370519)

    def next_batch(self, batch_size):
        self.batch = (self.batch + 1) % self.batches
        if self.batch == 0:
            self._seed()

        cfg = self.config
        sentence1 = np.random.randint(0, cfg.words1, [batch_size, cfg.num_steps1])
        sentence2 = np.random.randint(0, cfg.words2, [batch_size, cfg.num_steps2])
        return sentence1, sentence2


if __name__ == '__main__':
    cfg = MyConfig()
    cfg.from_cmd()
