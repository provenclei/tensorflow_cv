# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p69_poem_muti_dropout.py
@Description    :  
@CreateTime     :  2020/7/25 21:24
------------------------------------
@ModifyTime     :  使用多层 LSTM + Dropout 进行唐诗生成
"""
import p50_framework as myf
from p48_BufferDS import BufferDS
from p66_read_qts import QTS
import tensorflow as tf
import numpy as np


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.qts_path = './texts/qts.txt'
        self._ds = None

        self.num_units = 200
        self.num_steps = 32
        self.batch_size = 100
        self.keep_prob = 0.6
        self.lr = 0.0001

    def ds(self):
        self.make_ds()
        return self._ds

    def make_ds(self):
        if self._ds is None:
            qts = QTS(self.qts_path)
            self._ds = BufferDS(1000, qts, self.batch_size)
            self._num_chinese_chars = qts.get_num_chars()

    def num_chinese_chars(self):
        self.make_ds()
        return self._num_chinese_chars

    def get_ds_train(self):
        return self.ds()

    def get_ds_test(self):
        return self.get_ds_train()

    def get_name(self):
        return 'p69'

    def get_sub_tensors(self, gpu_id):
        return MySubTensors(self)

    def get_app(self):
        return MyApp(self)


class MySubTensors:
    def __init__(self, cfg: MyConfig):
        self.config = cfg

        x = tf.placeholder(tf.int64, [None, cfg.num_steps], 'x')
        self.inputs = [x]

        x = tf.one_hot(x, cfg.num_chinese_chars())  # [-1, 32, 4340]
        y = tf.layers.dense(x, cfg.num_units, name='dense1')  # [-1, 32, 200]

        cell1 = tf.nn.rnn_cell.LSTMCell(cfg.num_units, name='cell1', state_is_tuple=False)
        cell2 = tf.nn.rnn_cell.LSTMCell(cfg.num_units, name='cell2', state_is_tuple=False)

        cell = tf.nn.MultiRNNCell([cell1, cell2], state_is_tuple=False)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, cfg.keep_prob)

        state = cell.zero_state(tf.shape(y)[0], y.dtype)
        y_predicts = []
        losses = []
        with tf.variable_scope('for') as scope:
            for i in range(cfg.num_steps):
                # input, state  -> output, new_state
                yi, state = cell(y[:, i, :], state)  # [-1, 200]

                # 计算 loss
                logits = tf.layers.dense(yi, cfg.num_chinese_chars(), name='dense2')  # [-1, 4340]
                if i < cfg.num_steps - 1:
                    # y_predicts: [num_steps - 1, -1]
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=x[:, i+1, :], logits=logits)
                    losses.append(loss)

                    y_predict = tf.argmax(logits, axis=1)  # [-1]
                    y_predicts.append(y_predict)
                scope.reuse_variables()

        # y_predicts: [num_steps - 1, -1]
        self.precise = self.get_precise(y_predicts)
        self.losses = [tf.reduce_mean(losses)]
        self.y_predicts = tf.transpose(y_predicts, [1, 0])  # [-1, num_steps - 1]

        # call cell one time.（测试阶段需要使用的张量）
        # 测试阶段使模型一个输入，一个输出，而不是 32 个输出
        self.xi = tf.placeholder(tf.int64, [None], 'xi')  # [-1]
        self.zero_state = cell.zero_state(tf.shape(self.xi)[0], y.dtype)
        xi = tf.one_hot(self.xi, cfg.num_chinese_chars())
        tf.get_variable_scope().reuse_variables()
        xi = tf.layers.dense(xi, cfg.num_units, name='dense1')   # [-1, num_units]
        # tuple 没有形状，所以设置 state_is_tuple=False 并且有两层cell，所以需要四倍的 num_units 长度
        self.state = tf.placeholder(tf.float32, [None, 4 * cfg.num_units])
        with tf.variable_scope('for'):
            yi, self.new_state = cell(xi, self.state)
            yi = tf.layers.dense(yi, cfg.num_chinese_chars(), name='dense2')  # [-1, 4340]
        self.yi_predict = tf.argmax(yi, axis=1)  # [-1]

    def get_precise(self, y_predicts):
        '''
        计算精确度
        :param y_predicts: [num_steps - 1, -1]
        :return:
        '''
        xs = self.inputs[0]
        precises = []
        for i in range(0, self.config.num_steps-1):
            y_predict = y_predicts[i]   # [-1]
            x = xs[:, i+1]   # [-1]
            precise = tf.equal(y_predict, x)
            precise = tf.cast(precise, tf.float32)
            precises.append(precise)
        return tf.reduce_mean(precises)


class MyApp(myf.App):
    def after_epoch(self, epoch):
        super(MyApp, self).after_epoch(epoch)
        fd = self.get_feed_dict(self.config.get_ds_train())
        # 使用最后一个 GPU 计算精确度
        precise = self.session.run(self.ts.sub_ts[-1].precise, fd)
        print('Epoch %d: precise = %.6f' % (epoch, precise))

    def test1(self, ds):
        qts = self.config.ds().ds
        chars = qts.get_num_chars()
        # 随机生成
        xi = np.random.randint(0, chars, [5, 1])
        result = self._get_poems(xi)
        for i in range(len(result)):
            print(qts.get_chars(*list(result[i, :])))

    def _get_poems(self, x):
        '''
        从定义的张量中获取诗歌
        :param x: 指定形状的张量
        :return:
        '''
        # xi: [-1, chars]
        x = np.array(x)
        chars = np.shape(x)[1]
        xi = x[:, 0]
        result = [xi]
        ts_ = self.ts.sub_ts[-1]
        state = self.session.run(ts_.zero_state, {ts_.xi: xi})

        for i in range(self.config.num_steps - 1):
            fd = {ts_.xi: xi, ts_.state: state}
            xi, state = self.session.run([ts_.yi_predict, ts_.new_state], fd)  # [-1]
            if i < chars - 1:
                result.append(x[:, i+1])
            else:
                result.append(xi)
        # result: [num_steps, -1]
        result = np.transpose(result, [1, 0])  # [-1, num_steps]
        return result

    def make_poem(self, ch_str):
        qts = self.config.ds().ds
        x = [qts.get_ids(ch_str)]  # [1, -1]
        result = self._get_poems(x)
        for i in range(len(result)):
            print(qts.get_chars(*list(result[i, :])))

    def test(self, ds):
        while True:
            chars = input('> Please input Chinese chars such as "春", "中国":')
            if chars is None or len(chars) == 0:
                break
            self.make_poem(chars)


if __name__ == '__main__':
    cfg = MyConfig()
    cfg.from_cmd()
