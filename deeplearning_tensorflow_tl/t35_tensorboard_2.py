# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t35_tensorboard_2.py
@Description    :  
@CreateTime     :  2020/6/22 09:05
------------------------------------
@ModifyTime     :  
"""
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import tensorflow as tf


class Config:
    def __init__(self):
        self.save_path = './models/p35/mnist'
        self.data_path = '../deeplearning_tensorflow_p/MNIST_data'
        self.log_path = './logs'
        self.batch_size = 256
        self.lr = 0.01
        self.epoches = 10
        self.new_model = True


class Samples:
    def __init__(self, config: Config):
        self.config = config
        self.data = read_data_sets(config.data_path)

    def get_train_data(self):
        return self.data.test.next_batch(self.config.batch_size)

    def get_test_data(self):
        return self.data.test.next_batch(self.config.batch_size)

    def get_valid_data(self):
        return self.data.validation.next_batch(self.config.batch_size)

    @property
    def get_num_of_train(self):
        return self.data.train.num_examples

    @property
    def get_num_of_test(self):
        return self.data.test.num_examples

    @property
    def get_num_of_valid(self):
        return self.data.validation.num_examples


class Tensors:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.int32, [None], name='y')

        x = tf.reshape(self.x, [-1, 28, 28, 1])
        x = tf.layers.conv2d(x, 3, 16, 1, 'same', activation=tf.nn.relu)  # [-1, 28, 28, 16]
        x = tf.layers.conv2d(x, 3, 32, 2, 'same', activation=tf.nn.relu)  # [-1, 14, 14, 32]
        x = tf.layers.conv2d(x, 3, 64, 2, 'same')  # [-1, 7, 7, 64]

        x = tf.layers.flatten(x)
        logits = tf.layers.dense(x, 10)  # [-1, 10]
        y = tf.one_hot(self.y, 10)  # [-1, 10]
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits, axis=1)
        self.loss = tf.reduce_mean(self.loss)
        self.loss_summary = tf.summary.scalar('loss', self.loss)

        y_predict = tf.argmax(logits, axis=1, output_type=tf.int32)
        self.precise = tf.cast(tf.equal(self.y, y_predict), tf.float32)
        self.precise = tf.reduce_mean(self.precise)
        self.precise_summary = tf.summary.scalar('precise', self.precise)

        self.lr = tf.placeholder(tf.float32, (), name='lr')
        opt = tf.train.AdamOptimizer(self.lr)
        self.opt_train = opt.minimize(self.loss)


class MNISTAPP:
    def __init__(self):
        self.session = tf.Session()
        self.ts = Tensors()
        self.config = Config()
        self.samples = Samples(self.config)
        self.saver = tf.train.Saver()

        try:
            self.saver.restore(self.session, self.config.save_path)
            print('模型恢复成功')
        except:
            self.session.run(tf.global_variables_initializer())
            print('模型初始化成功')

    def train(self):
        writer = tf.summary.FileWriter(self.config.log_path, graph=self.session.graph)
        batches = self.samples.get_num_of_test // self.config.batch_size
        for epoche in range(self.config.epoches):
            for batch in range(batches):
                xs_train, ys_train = self.samples.get_test_data()
                _, loss, loss_summary = self.session.run([self.ts.opt_train, self.ts.loss, self.ts.loss_summary], {
                    self.ts.x: xs_train,
                    self.ts.y: ys_train,
                    self.ts.lr: self.config.lr
                })
                writer.add_summary(loss_summary, epoche*batches+batch)
                print('epoch: %d ---- batch: %d ---- loss: %f ' % (epoche, batch, loss))
            # 验证
            xs_valid, ys_valid = self.samples.get_valid_data()
            precise, precise_summary = self.session.run([self.ts.precise, self.ts.precise_summary], {
                self.ts.x: xs_valid,
                self.ts.y: ys_valid
            })
            writer.add_summary(precise_summary)
            print('验证集准确度：%f' % precise)
            self.save()
        writer.close()

    def predict(self):
        batches = self.samples.get_num_of_test // self.config.batch_size
        total_precise = 0.0
        for batch in range(batches):
            xs_test, ys_test = self.samples.get_test_data()
            precise = self.session.run(self.ts.precise, {
                self.ts.x: xs_test,
                self.ts.y: ys_test
            })
            total_precise += precise
        print('total_precise: %f' % (total_precise/batches))

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('模型保存成功')

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    app = MNISTAPP()
    with app:
        # app.train()
        app.predict()


if __name__ == '__main__':
    main()