# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p32_sin.py
@Description    :  
@CreateTime     :  2020/6/16 10:34
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Config:
    '''
    参数设置
    '''
    def __init__(self):
        self.save_path = './models/p32/sin'
        self.lr = 0.01
        self.epoches = 500
        # self.batch_size = 200
        self.samples = 200
        self.hidden_units = 200


class Sample:
    '''
    数据预处理
    '''
    def __init__(self, samples):
        self.xs = np.random.uniform(-np.pi, np.pi, [samples])
        self.xs = sorted(self.xs)
        self.ys = np.sin(self.xs)
        # plt.plot(xs, ys)
        # plt.show()

    @property
    def num_examples(self):
        return len(self.xs)


class Tensors:
    '''
    存储变量
    '''
    def __init__(self, config: Config):
        self.x = tf.placeholder(tf.float32, [None], 'x')
        self.y = tf.placeholder(tf.float32, [None], 'y')

        x = tf.reshape(self.x, [-1, 1])
        x = tf.layers.dense(x, config.hidden_units, tf.nn.relu)
        y = tf.layers.dense(x, 1)  # [-1, 1]

        self.y_predict = tf.reshape(y, [-1])  # [-1]
        self.loss = tf.reduce_mean(tf.square(self.y_predict - self.y))
        self.lr = tf.placeholder(tf.float32, [], 'lr')
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = opt.minimize(self.loss)
        self.loss = tf.sqrt(self.loss)


class SinApp:
    def __init__(self, config: Config):
        self.config = config
        self.ts = Tensors(config)
        self.session = tf.Session()
        self.saver = tf.train.Saver()

        try:
            # 恢复模型
            self.saver.restore(self.session, config.save_path)
            print('Restore model from %s successfully!' % config.save_path)
        except:
            print('Fail to restore the model from %s, use a new empty one.' % config.save_path)
            self.session.run(tf.global_variables_initializer())

    def train(self):
        sample = Sample(self.config.samples)
        cfg = self.config
        ts = self.ts
        for epoch in range(cfg.epoches):
            _, loss = self.session.run([ts.train_op, ts.loss],
                                       {ts.x: sample.xs, ts.y: sample.ys, ts.lr: cfg.lr})
            print('%d, loss = %.6f' % (epoch, loss))
        self.save()
        return sample.xs, sample.ys

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('Save model into', self.config.save_path)

    def predict(self):
        sample = Sample(400)
        ys = self.session.run(self.ts.y_predict, {
            self.ts.x: sample.xs
        })
        return sample.xs, ys

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    app = SinApp(Config())
    with app:
        xs_train, ys_train = app.train()
        xs_predict, ys_predict = app.predict()
    plt.plot(xs_train, ys_train)
    plt.plot(xs_predict, ys_predict)
    plt.show()


if __name__ == '__main__':
    main()