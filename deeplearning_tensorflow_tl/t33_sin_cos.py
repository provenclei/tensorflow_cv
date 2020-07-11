# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t33_sin_cos.py
@Description    :  
@CreateTime     :  2020/6/18 18:00
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Config:
    def __init__(self):
        self.save_path = './models/p33/2/sin_cos'
        self.batch_size = 256
        self.hhidden_layers = 200
        self.lr = 0.01
        self.samples = 1000
        self.epoches = 500


class Tensors:
    def __init__(self, config: Config):
        self.x = tf.placeholder(tf.float32, [None], name='x')
        x = tf.reshape(self.x, [-1, 1])  # [-1, 1]
        x = tf.layers.dense(x, config.hhidden_layers, tf.nn.leaky_relu)  # [-1, 100]
        self.y_predict = tf.layers.dense(x, 3)  # [-1, 3]

        self.y = tf.placeholder(tf.float32, [3, None], name='y')   # [3, -1]
        y = tf.transpose(self.y)  # [-1, 3]
        self.loss = tf.reduce_mean(tf.square(self.y_predict - y))

        self.lr = tf.placeholder(tf.float32, (), name='lr')
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_opt = opt.minimize(self.loss)


class Samples:
    def __init__(self, samples):
        self.xs = np.random.uniform(-np.pi, np.pi, [samples])
        self.xs = sorted(self.xs)
        self.ys = np.sin(self.xs), np.cos(self.xs), np.square(self.xs)


class SINApp:
    def __init__(self, config: Config):
        self.config = config
        self.session = tf.Session()
        self.ts = Tensors(config)
        self.saver = tf.train.Saver()

        try:
            self.saver.restore(self.session, config.save_path)
            print('Restore model from %s successfully!' % config.save_path)
        except:
            print('Fail to restore the model from %s, use a new empty one.' % config.save_path)
            self.session.run(tf.global_variables_initializer())

    def train(self):
        samples = Samples(self.config.samples)
        for epoch in range(self.config.epoches):
            _, loss = self.session.run([self.ts.train_opt, self.ts.loss], {
                self.ts.x: samples.xs,
                self.ts.y: samples.ys,
                self.ts.lr: self.config.lr
            })
            print('%d, loss = %.6f' % (epoch, loss))
        self.save()
        return samples.xs, samples.ys

    def predict(self):
        sample = Samples(500)
        ys = self.session.run(self.ts.y_predict, {
            self.ts.x: sample.xs
        })
        return sample.xs, ys

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('Save model into', self.config.save_path)

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    config = Config()
    app = SINApp(config)
    with app:
        xs_train, ys_train = app.train()
        xs_predict, ys_predict = app.predict()
    ys_train = np.transpose(ys_train)
    plt.plot(xs_train, ys_train)
    plt.plot(xs_predict, ys_predict)
    plt.show()


if __name__ == '__main__':
    main()