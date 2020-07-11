# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t30_sample_sqrt.py
@Description    :  
@CreateTime     :  2020/6/14 12:31
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SQRTAPP:
    def __init__(self):
        x = tf.placeholder(tf.float32, [None], 'x')
        self.x = x
        x = tf.reshape(x, [-1, 1])
        w = tf.get_variable('w', [1, 200], tf.float32)
        b = tf.get_variable('b', [200], tf.float32)
        x = tf.matmul(x, w) + b  # [-1, 200]
        x = tf.nn.relu(x)  # relu(x) === maximum(x, 0)

        w = tf.get_variable('w2', [200, 1], tf.float32)
        b = tf.get_variable('b2', [1], tf.float32)
        self.y_predict = tf.matmul(x, w) + b  # [-1, 1]
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

    def sqrt(self, xp):
        ys = self.session.run(self.y_predict, {self.x: xp})
        return ys

    def close(self):
        self.session.close()


def main():
    ys = np.array([e for e in range(0, 200)]) / 100
    xs = np.square(ys)
    plt.plot(xs, ys)

    # 训练
    app = SQRTAPP()
    app.train(xs, ys)

    # 测试
    xp = np.random.uniform(0, 2, [400])
    xp = sorted(xp)
    yp = app.sqrt(xp)
    plt.plot(xp, yp)

    app.close()
    plt.show()


if __name__ == '__main__':
    main()