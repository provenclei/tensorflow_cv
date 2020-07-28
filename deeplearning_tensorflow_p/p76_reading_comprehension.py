# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p76_reading_comprehension.py
@Description    :  
@CreateTime     :  2020/7/28 16:47
------------------------------------
@ModifyTime     :  使用 framework 和 self-attention 实现阅读理解
"""
import tensorflow as tf
import numpy as np
import p74_framework as myf
from p73_self_attention import self_attention


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.num_steps1 = 150
        self.num_steps2 = 20
        self.num_steps3 = 20
        self.num_units = 200
        self.words = 6000
        self.keep_prob = 0.6
        self._ds = None
        self.batch_size = 100
        self.lr = 0.0001

    def get_name(self):
        return 'p76'

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
        self.reading = tf.placeholder(tf.int64, [None, cfg.num_steps1], 'reading')
        self.question = tf.placeholder(tf.int64, [None, cfg.num_steps2], 'question')
        self.answer = tf.placeholder(tf.int64, [None, cfg.num_steps2], 'answer')
        self.inputs = [self.reading, self.question, self.answer]

        dict1 = tf.get_variable('dict1', [cfg.words, cfg.num_units], tf.float64)  # 单词向量字典  [6000, 200]
        r = tf.nn.embedding_lookup(dict1, self.reading)   # [-1, steps1, units], use word2vec instead of one-hot and dense

        cell1 = [tf.nn.rnn_cell.LSTMCell(cfg.num_units, name='cell1_%d' % i) for i in range(2)]
        cell1 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(cell1), cfg.keep_prob)

        # read
        batch_size = tf.shape(r)[0]
        state = cell1.zero_state(batch_size, r.dtype)
        read_out = []
        for i in range(cfg.num_steps1):
            out, state = cell1(r[:, i, :], state)
            read_out.append(out)
        # read_out: [steps1, -1, units]

        # question
        cell2 = [tf.nn.rnn_cell.LSTMCell(cfg.num_units, name='cell2_%d' % i) for i in range(2)]
        cell2 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(cell2), cfg.keep_prob)
        q = tf.nn.embedding_lookup(dict1, self.question)
        question_out = []
        for i in range(cfg.num_steps2):
            out, state = cell2(q[:, i, :], state)
            question_out.append(out)

        att_vec = self_attention(read_out, cfg.num_steps3, 'self_attention1')  # [steps3, -1, units]
        att_vec += self_attention(question_out, cfg.num_steps3, 'self_attention2')  # [steps3, -1, units]

        # answer
        cell3 = [tf.nn.rnn_cell.LSTMCell(cfg.num_units, name='cell3_%d' % i) for i in range(2)]
        cell3 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(cell3), cfg.keep_prob)
        y_predict = []
        loss = []
        a = tf.one_hot(self.answer, cfg.words)  # [-1, steps2, words]
        for i in range(cfg.num_steps3):
            yi, state = cell3(att_vec[i, :, :], state)  # [-1, units]

            logits = tf.layers.dense(yi, cfg.words, name='dense1')  # [-1, words]
            tf.get_variable_scope().reuse_variables()

            yi_predict = tf.argmax(logits, axis=1)  # [-1]
            y_predict.append(yi_predict)

            loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=a[:, i, :], logits=logits)
            loss.append(loss_i)

        # y: [steps2, -1]
        self.loss = tf.reduce_mean(loss)
        self.y_predict = tf.transpose(y_predict)  # [-1, steps3]
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
        sentence1 = np.random.randint(0, cfg.words, [batch_size, cfg.num_steps1])
        sentence2 = np.random.randint(0, cfg.words, [batch_size, cfg.num_steps2])
        sentence3 = np.random.randint(0, cfg.words, [batch_size, cfg.num_steps3])
        return sentence1, sentence2, sentence3


if __name__ == '__main__':
    cfg = MyConfig()
    cfg.from_cmd()
