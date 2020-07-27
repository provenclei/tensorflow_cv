# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p70_bidirec_RNN.py
@Description    :  
@CreateTime     :  2020/7/26 18:42
------------------------------------
@ModifyTime     :  实现双向 LSTM
                使用的局限性：需要从左到右或者从右到左都已知的序列文本任务
                比如：翻译，阅读理解，多轮对话，分词
"""
import tensorflow as tf


name_id = 1


def bidirec_rnn(cell_l2r, cell_r2l, x, state_is_tuple=True, name=None):
    # x: [-1, num_steps, num_units]
    if name is None:
        global name_id
        name = 'bidirec_rnn_%d' % name_id
        name_id += 1

    with tf.variable_scope(name):
        num_steps = x.shape[1].value

        batch_size = tf.shape(x)[0]
        state_l2r = cell_l2r.zero_state(batch_size, x.dtype)
        state_r2l = cell_r2l.zero_state(batch_size, x.dtype)
        y_l2r = []
        y_r2l = []
        for i in range(num_steps):
            yi_l2r, state_l2r = cell_l2r(x[:, i, :], state_l2r)  # [-1, num_units]
            yi_r2l, state_r2l = cell_r2l(x[:, num_steps-1-i, :], state_r2l)
            y_l2r.append(yi_l2r)
            # 插入到第一个位置，使其对应起来
            y_r2l.insert(0, yi_r2l)

        # yi_l2r， yi_r2l -> [num_steps, -1, num_units]
        y = [yi_l2r + yi_r2l for yi_l2r, yi_r2l in zip(y_l2r, y_r2l)]  # [num_steps, -1, num_units]
        y = tf.transpose(y, [1, 0, 2])  # [-1, num_steps, num_units]
        state = (state_l2r, state_r2l) if state_is_tuple else tf.concat((state_l2r, state_r2l), axis=1)
        return y, state


if __name__ == '__main__':
    cell1 = tf.nn.rnn_cell.LSTMCell(200, name='cell1')
    cell2 = tf.nn.rnn_cell.LSTMCell(200, name='cell2')

    x = tf.random_normal([123, 32, 200])
    y, state = bidirec_rnn(cell1, cell2, x)
    print(y.shape)
    print(state[0])
    print(state[1])
