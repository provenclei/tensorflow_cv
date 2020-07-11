# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t31_mnist_2.py
@Description    :  
@CreateTime     :  2020/6/15 15:57
------------------------------------
@ModifyTime     :  
"""
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import tensorflow as tf


class Tensors:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784], 'x')
        x = tf.layers.dense(self.x, 2000, tf.nn.relu)
        x = tf.layers.dense(x, 10)  # [-1, 10]
        y_logic = tf.nn.softmax(x)
        y_predict = tf.maximum(y_logic, 1e-6)

        self.y = tf.placeholder(tf.int32, [None], 'y')
        y = tf.one_hot(self.y, 10)

        self.loss = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_predict), axis=1))
        self.lr = tf.placeholder(tf.float32, (), name='lr')
        op = tf.train.AdamOptimizer(self.lr)
        self.train_opt = op.minimize(self.loss)


class MNISTAPP:
    def __init__(self):
        self.ts = Tensors()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def train(self, data, lr_s=0.01, epoches=2000, batch_size=200):
        for epoch in range(epoches):
            for batch in range(batch_size):
                xs, ys = data.train.next_batch(batch_size)
                _, loss = self.session.run([self.ts.train_opt, self.ts.loss], \
                                 {self.ts.lr: lr_s, self.ts.x: xs, self.ts.y: ys})
                print('epoch: %d-------batch: %d------loss: %.5f' % (epoch, batch, loss))

    def predict(self):
        pass


def main():
    path = '../artifitial_intelligent_p.MNIST_data'
    data = read_data_sets(path)
    app = MNISTAPP()
    app.train(data)
    app.predict()


if __name__ == '__main__':
    main()