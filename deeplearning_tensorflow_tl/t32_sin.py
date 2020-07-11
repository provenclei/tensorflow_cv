# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t32_sin.py
@Description    :  
@CreateTime     :  2020/6/16 17:55
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Config:
    def __init__(self):
        self.lr = 0.01
        self.epoches = 500
        self.hiden_states = 1000
        self.save_path = './models/p32/1/sin'
        self.samples = 2000


class Tensors:
    def __init__(self, config: Config):
        self.x = tf.placeholder(tf.float32, [None], name='x')
        self.y = tf.placeholder(tf.float32, [None], name='y')

        x = tf.reshape(self.x, [-1, 1])
        x = tf.layers.dense(x, config.hiden_states, tf.nn.relu)
        y = tf.layers.dense(x, 1)  # [-1, 1]

        self.y_predict = tf.reshape(y, [-1])
        self.loss = tf.reduce_mean(tf.square(self.y - self.y_predict))
        self.lr = tf.placeholder(tf.float32, (), 'lr')
        opt = tf.train.AdamOptimizer(self.lr)
        self.opt_train = opt.minimize(self.loss)
        # self.loss = tf.sqrt(self.loss)


class Sample:
    def __init__(self, samples):
        xs_train = np.random.uniform(-np.pi, np.pi, [samples])
        self.xs_train = sorted(xs_train)
        self.ys_train = np.sin(self.xs_train)

        # xs_test = np.random.uniform(-np.pi, np.pi, [samples])
        # self.xs_test = sorted(xs_test)
        # self.ys_test = np.sin(self.xs_test)

    def num_examples(self):
        return len(self.xs_train)


class SinApp:
    def __init__(self, config: Config):
        self.config = config
        self.session = tf.Session()
        self.ts = Tensors(config)
        self.saver = tf.train.Saver()

        try:
            self.saver.save(self.session, config.save_path)
            print('Restore model from %s successfully!' % config.save_path)
        except:
            print('Fail to restore the model from %s, use a new empty one.' % config.save_path)
            self.session.run(tf.global_variables_initializer())

    def train(self):
        sample = Sample(self.config.samples)
        for epoch in range(self.config.epoches):
            _, loss = self.session.run([self.ts.opt_train, self.ts.loss], {
                self.ts.x: sample.xs_train,
                self.ts.y: sample.ys_train,
                self.ts.lr: self.config.lr
            })
            print('%d, loss = %.6f' % (epoch, loss))
        self.save()
        return sample.xs_train, sample.ys_train

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('Save model into', self.config.save_path)

    def predict(self):
        sample = Sample(400)
        ys = self.session.run(self.ts.y_predict, {
            self.ts.x: sample.xs_train
        })
        return sample.xs_train, ys

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