# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p32_mnist_3.py
@Description    :  
@CreateTime     :  2020/6/15 14:33
------------------------------------
@ModifyTime     :  增加 predict 和 validation
                    计算精确度
                        保存模型
"""
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import tensorflow as tf


class Tensor():
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        x = tf.layers.dense(self.x, 2000, tf.nn.relu)  # [-1, 2000]
        logits = tf.layers.dense(x, 10)  # [-1, 10]

        # 预测值
        self.y_predict = tf.argmax(logits, axis=1, output_type=tf.int32)  # [-1]
        # 获得预测概率
        # 为了避免交叉熵损失函数中log(0)发生
        y_pred = tf.maximum(tf.nn.softmax(logits), 1e-6)  # [-1, 10] 概率

        self.y = tf.placeholder(tf.int32, [None], name='y')
        y = tf.one_hot(self.y, 10)

        self.loss = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_pred), axis=1))
        self.lr = tf.placeholder(tf.float32, (), name='lr')
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_opt = opt.minimize(self.loss)

        precise = tf.cast(tf.equal(self.y, self.y_predict), tf.float32)
        self.precise = tf.reduce_mean(precise)


class MNISTApp():
    def __init__(self, save_path):
        self.save_path = save_path
        self.ts = Tensor()
        self.session = tf.Session()
        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.session, save_path)
            print('restore model from %s successfully!' % save_path)
        except:
            print('fail to restore model from %s, use a new empty model instead' % save_path)
            # 空模型替代
            self.session.run(tf.global_variables_initializer())

    def train(self, data, lr_s=0.01, epoches=50, batch_size=300):
        for epoche in range(epoches):
            batches = data.train.num_examples // batch_size
            for batch in range(batches):
                xs, ys = data.train.next_batch(batch_size)
                _, loss = \
                    self.session.run([self.ts.train_opt, self.ts.loss], \
                                     {self.ts.x: xs, self.ts.y: ys, self.ts.lr: lr_s})
                print('%d, %d, loss:%.5f' % (epoche, batch, loss))
            self.valid(data, batch_size)
            self.save()

    def save(self):
        self.saver.save(self.session, self.save_path)
        print('save model into:', self.save_path)

    def valid(self, data, batch_size=300):
        xs, ys = data.validation.next_batch(batch_size)
        # ys_predict = self.session.run(self.ts.y_predict, {self.ts.x: xs})
        # print('predict:\t', ys_predict)
        precise = self.session.run(self.ts.precise, {self.ts.x: xs, self.ts.y: ys})
        print('precise:\t', precise)

    def predict(self, data, batch_size=300):
        xs, ys = data.test.next_batch(batch_size)
        ys_predict = self.session.run(self.ts.y_predict, {self.ts.x: xs})
        print('real:\t', ys)
        print('predict:\t', ys_predict)

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    path = './MNIST_data'
    data = read_data_sets(path)
    print(data.train.num_examples, data.test.num_examples, data.validation.num_examples)

    # mnist表示前缀
    save_path = './models/p31/mnist'
    app = MNISTApp(save_path)
    # 方法一
    # try:
    #     app.train(data)
    #     app.predict(data)
    # finally:
    #     app.close()

    # 方法二:
    with app:
        app.train(data)
        app.predict(data)


if __name__ == '__main__':
    main()