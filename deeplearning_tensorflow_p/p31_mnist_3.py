# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p32_mnist_3.py
@Description    :  
@CreateTime     :  2020/6/15 14:33
------------------------------------
@ModifyTime     :  
"""
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import tensorflow as tf


class Tensor:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')   # [-1, 784]
        x = tf.layers.dense(self.x, 2000, tf.nn.relu)  # [-1, 2000]
        x = tf.layers.dense(x, 10)  # [-1, 10]
        y_prob = tf.nn.softmax(x)  # [-1, 10]
        # 为了避免交叉熵损失函数中log(0)发生
        y_predict = tf.maximum(y_prob, 1e-6)  # [-1, 10]

        self.y = tf.placeholder(tf.int32, [None], name='y')    # [-1]
        y = tf.one_hot(self.y, 10)  # [-1, 10]

        self.loss = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_predict), axis=1))
        self.lr = tf.placeholder(tf.float32, (), name='lr')
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_opt = opt.minimize(self.loss)


class MNISTApp:
    def __init__(self):
        self.ts = Tensor()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def train(self, data, lr_s=0.01, epoches=20, batch_size=200):
        for epoche in range(epoches):
            batches = data.train.num_examples // batch_size
            for batch in range(batches):
                xs, ys = data.train.next_batch(batch_size)
                _, loss = \
                    self.session.run([self.ts.train_opt, self.ts.loss], \
                                     {self.ts.x: xs, self.ts.y: ys, self.ts.lr: lr_s})
                print('%d, %d, loss:%.5f' % (epoche, batch, loss))

    def predict(self):
        pass


def main():
    path = './MNIST_data'
    data = read_data_sets(path)
    print(data.train.num_examples, data.test.num_examples, data.validation.num_examples)
    app = MNISTApp()
    app.train(data)


if __name__ == '__main__':
    main()