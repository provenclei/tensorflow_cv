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
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Config:
    '''
    参数设置
    '''
    def __init__(self):
        self.save_path = './models/p34/conv_mnist'
        self.sample_path = './MNIST_data'
        self.lr = 0.01
        self.epoches = 1000
        self.batch_size = 200


class Tensors:
    '''
    存储变量
    '''
    def __init__(self, config: Config):
        self.x = tf.placeholder(tf.float32, [None, 784], 'x')
        self.y = tf.placeholder(tf.int32, [None], 'y')

        x = tf.reshape(self.x, [-1, 28, 28, 1])
        # channel, filter, feature
        x = tf.layers.conv2d(x, filters=16, kernel_size=(3, 3), strides=1, padding='same', activation=tf.nn.relu)  # [-1, 28, 28, 16]
        x = tf.layers.conv2d(x, 32, 3, 2, 'same', activation=tf.nn.relu)  # [-1, 14, 14, 32]
        x = tf.layers.conv2d(x, 64, 3, 2, 'same', activation=tf.nn.relu)  # [-1, 7, 7, 64]

        x = tf.layers.flatten(x)   # [-1, 7*7*64]
        logits = tf.layers.dense(x, 10)

        y = tf.one_hot(self.y, 10)
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
        self.loss = tf.reduce_mean(self.loss)

        self.y_predict = tf.argmax(logits, axis=1, output_type=tf.int32)  # [-1]
        self.precise = tf.cast(tf.equal(self.y, self.y_predict), tf.float32)
        self.precise = tf.reduce_mean(self.precise)

        self.lr = tf.placeholder(tf.float32, (), 'lr')
        opt = tf.train.AdamOptimizer(self.lr)
        self.opt_train = opt.minimize(self.loss)


class SinApp:
    def __init__(self, config: Config, data):
        self.config = config
        self.ts = Tensors(config)
        self.session = tf.Session()
        self.saver = tf.train.Saver()
        self.data = data

        try:
            # 恢复模型
            self.saver.restore(self.session, config.save_path)
            print('Restore model from %s successfully!' % config.save_path)
        except:
            print('Fail to restore the model from %s, use a new empty one.' % config.save_path)
            self.session.run(tf.global_variables_initializer())

    def train(self):
        cfg = self.config
        ts = self.ts
        for epoch in range(cfg.epoches):
            # MBGD
            batches = self.data.train.num_examples // self.config.batch_size
            for batch in range(batches):
                xs, ys = self.data.train.next_batch(self.config.batch_size)
                _, loss = self.session.run([ts.opt_train, ts.loss],
                                           {ts.x: xs, ts.y: ys, ts.lr: self.config.lr})
                print('%d/%03d. loss = %.6f' % (epoch, batch, loss))
            xs, ys = self.data.validation.next_batch(self.config.batch_size)
            precise = self.session.run(ts.precise,
                                                {ts.x: xs, ts.y: ys})
            # linux中打印存在缓冲区，不会马上刷新在屏幕上
            print('%d, precise:%.6f', (epoch, precise), flush=True)
            self.save()

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('Save model into', self.config.save_path)

    def predict(self, data, batch_size=200):
        precise_total = 0
        batches = data.test.num_examples // batch_size
        for _ in range(batches):
            xs, ys = data.test.next_batch(batch_size)
            precise = self.session.run(self.ts.precise, {
                self.ts.x: xs,
                self.ts.y: ys
            })
            precise_total += precise
        print('precise_total:', precise_total/batch_size)

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    config = Config()
    data = read_data_sets(config.sample_path)
    app = SinApp(config, data)
    with app:
        xs_train, ys_train = app.train()
        xs_predict, ys_predict = app.predict()
        ys_train = np.transpose(ys_train)
        plt.plot(xs_train, ys_train)
        plt.plot(xs_predict, ys_predict)
        plt.show()


if __name__ == '__main__':
    main()