# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p30_sample_sqrt_2.py
@Description    :  
@CreateTime     :  2020/6/13 18:51
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# class SqrtApp:
#     def __init__(self):
#         # [-1, 1]
#         self.x = tf.placeholder(tf.float32, [None], name='x')
#         self.x = tf.reshape(self.x, [-1, 1])
#
#         w = tf.get_variable('w', [1, 200], tf.float32)
#         b = tf.get_variable('b', [200], tf.float32)
#         self.x = tf.matmul(self.x, w) + b
#         self.x = tf.nn.relu(self.x)  # [-1, 200]
#
#         w = tf.get_variable('w2', [200, 1], tf.float32)
#         b = tf.get_variable('b2', [1], tf.float32)
#         self.y_pre = tf.matmul(self.x, w) + b  # [-1, 1]
#         self.y_pre = tf.reshape(self.y_pre, [-1])
#
#         self.y = tf.placeholder(tf.float32, [None], 'y')
#         loss = tf.reduce_mean(tf.square(self.y - self.y_pre))
#         self.lr = tf.placeholder(tf.float32, [], name='lr')
#         opt = tf.train.GradientDescentOptimizer(self.lr)
#         self.opt_train = opt.minimize(loss)
#
#         self.session = tf.Session()
#         self.session.run(tf.global_variables_initializer())

#     def train(self, xs, ys, lr_s=0.01, epoches=2000):
#         for _ in range(epoches):
#             self.session.run(self.opt_train, {self.x: xs, self.y: ys, self.lr: lr_s})
#
#     def sqrt(self, xs):
#         return self.session.run(self.y_pre, {self.x: xs})
#
#     def close(self):
#         self.session.close()
#
#
# def main():
#     ys = np.array([e for e in range(0, 200)])/100
#     xs = np.sqrt(ys)
#     print(xs.shape, ys.shape)
#     app = SqrtApp()
#     app.train(xs, ys)
#     plt.plot(xs, ys)
#
#     app.close()
#     plt.show()
#
#     # for i in range(2, 11):
#     #     print('sqrt(%s) = %.6f' % (i, app.sqrt(i)))
#     # app.close()


# if __name__ == '__main__':
#     main()


class SqrtApp:
    def __init__(self):
        x = tf.placeholder(tf.float32, [None], 'x')
        self.x = x
        x = tf.reshape(x, [-1, 1])
        w = tf.get_variable('w', [1, 200], tf.float32)
        b = tf.get_variable('b', [200], tf.float32)
        x = tf.matmul(x, w) + b  # [-1, 200]
        x = tf.nn.relu(x)    # relu(x) === maximum(x, 0)

        w = tf.get_variable('w2', [200, 1], tf.float32)
        b = tf.get_variable('b2', [1], tf.float32)
        self.y_predict = tf.matmul(x, w) + b    # [-1, 1]
        self.y_predict = tf.reshape(self.y_predict, [-1])

        self.y = tf.placeholder(tf.float32, [None], 'y')
        loss = tf.reduce_mean(tf.square(self.y - self.y_predict))
        self.lr = tf.placeholder(tf.float32, [], 'lr')
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = opt.minimize(loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def train(self, xs, ys, lr=0.01, epoches=2000):
        for _ in range(epoches):
            self.session.run(self.train_op, {self.x: xs, self.y: ys, self.lr: lr})

    def sqrt(self, xs):
        ys = self.session.run(self.y_predict, {self.x: xs})
        return ys

    def close(self):
        self.session.close()


if __name__ == '__main__':
    app = SqrtApp()
    ys = np.array([e for e in range(0, 200)]) / 100
    xs = np.square(ys)
    # 训练
    app.train(xs, ys)
    plt.plot(xs, ys)

    xs = np.random.uniform(0, 2, [400])
    xs = sorted(xs)
    # 测试
    ys = app.sqrt(xs)
    plt.plot(xs, ys)

    app.close()
    plt.show()
