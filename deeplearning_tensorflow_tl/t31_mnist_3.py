# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t31_mnist_3.py
@Description    :  
@CreateTime     :  2020/6/16 15:02
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets


class Config:
    def __init__(self):
        self.lr = 0.01
        self.batch_size = 300
        self.save_path = './models/p31/mnist'
        self.data_path = '../deeplearning_tensorflow_p/MNIST_data'
        self.epoches = 500
        self.hidden_states_1 = 2000
        self.hidden_states_2 = 10
        self.input_dim = 784


class Tensors:
    def __init__(self, config: Config):
        x = tf.placeholder(tf.float32, [None, config.input_dim], name='x')
        self.x = x
        x = tf.layers.dense(self.x, config.hidden_states_1, tf.nn.relu)
        logits = tf.layers.dense(x, config.hidden_states_2)   # [-1, 10]
        self.y_predict = tf.argmax(logits, axis=1, output_type=tf.int32)  # [-1]
        y_prob = tf.maximum(tf.nn.softmax(logits), 1e-6)   # [-1, 10]

        self.y = tf.placeholder(tf.int32, [None], name='y')  # [-1]
        y = tf.one_hot(self.y, 10)  # [-1, 10]
        self.loss = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_prob), axis=1))
        opt = tf.train.AdamOptimizer(config.lr)
        self.opt_train = opt.minimize(self.loss)

        precise = tf.cast(tf.equal(self.y, self.y_predict), tf.float32)
        self.precise = tf.reduce_mean(precise)


class MMISTApp:
    def __init__(self, config: Config):
        self.session = tf.Session()
        self.tensors = Tensors(config)
        self.config = config
        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.session, self.config.save_path)
            print('restore model from %s successfully!' % self.config.save_path)
        except:
            self.session.run(tf.global_variables_initializer())
            print('fail to restore model from %s, use a new empty model instead' % self.config.save_path)

    def train(self, data):
        for epoche in range(self.config.epoches):
            batches = data.train.num_examples // self.config.batch_size
            for batch in range(batches):
                xs, ys = data.train.next_batch(self.config.batch_size)
                _, loss = \
                    self.session.run([self.tensors.opt_train, self.tensors.loss], {
                        self.tensors.x: xs, self.tensors.y: ys
                })
                print('%d, %d, loss:%.5f' % (epoche, batch, loss))
            self.valid(data)
            self.save()

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('save model into:', self.config.save_path)

    def predict(self, data):
        xs, ys = data.test.next_batch(self.config.batch_size)
        y_predict = self.session.run(self.tensors.y_predict, {
            self.tensors.x: xs
        })
        print('真实值：\t', ys)
        print("预测值：\t", y_predict)

    def valid(self, data):
        xs, ys = data.validation.next_batch(self.config.batch_size)
        precise = self.session.run(self.tensors.precise, {
            self.tensors.x: xs,
            self.tensors.y: ys
        })
        print('precise:\t', precise)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()


def main():
    config = Config()
    data = read_data_sets(config.data_path)
    print(data.train.num_examples,
          data.test.num_examples,
          data.validation.num_examples)

    app = MMISTApp(config)
    with app:
        app.train(data)
        app.predict(data)


if __name__ == '__main__':
    main()