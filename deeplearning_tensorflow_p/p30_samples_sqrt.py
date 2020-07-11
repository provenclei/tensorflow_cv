# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p30_samples_sqrt.py
@Description    :  
@CreateTime     :  2020/6/12 10:44
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class SqrtApp:
    def __init__(self):
        # x = tf.placeholder(tf.float32, [None], name='x')
        # 只有在 placeholder 中使用none表示-1
        # 等价于[-1, 1] 或 [-1]
        x = tf.placeholder(tf.float32, [None], name='x')
        self.x = x
        x = tf.reshape(x, [-1, 1])

        w = tf.get_variable('w', [1, 200], tf.float32)
        b = tf.get_variable('b', [200], tf.float32)
        x = tf.matmul(x, w) + b  # [-1, 200]
        x = tf.nn.relu(x)

        w = tf.get_variable('w2', [200, 1], tf.float32)
        b = tf.get_variable('b2', [1], tf.float32)
        self.y_predict = tf.matmul(x, w) + b  # [-1, 1]
        self.y_predict = tf.reshape(self.y_predict, [-1])

        self.y = tf.placeholder(tf.float32, [None], 'y')
        loss = tf.reduce_mean(tf.square(self.y - self.y_predict))
        self.lr = tf.placeholder(tf.float32, [], name='lr')
        opt = tf.train.GradientDescentOptimizer(self.lr)
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


def main():
    ys = np.array([e for e in range(0, 200)])/100
    xs = np.square(ys)

    app = SqrtApp()
    # app.train(xs, ys)
    plt.plot(xs, ys)

    xs = np.random.uniform(0, 2, [400])
    xs = sorted(xs)
    ys = app.sqrt(xs)
    plt.plot(xs, ys)
    plt.show()

    app.close()


if __name__ == '__main__':
    main()