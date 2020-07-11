# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t31_mnist_4.py
@Description    :  
@CreateTime     :  2020/6/17 10:18
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
'''
当有多个输出时，做个输出相互之间不干扰，相互独立
'''


class Config:
    def __init__(self):
        self.data_path = '../deeplearning_tensorflow_p/MNIST_data'
        self.save_path = './models/p32/3/mnist'
        self.batch_size = 300
        self.hidden_state_1 = 200
        self.hidden_state_2 = 10
        self.epoches = 500
        self.lr = 0.01


class Samples:
    def __init__(self, config: Config):
        self.data = read_data_sets(config.data_path)


class Tensors:
    def __init__(self, config: Config):
        x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.x = x
        x = tf.layers.dense(x, config.hidden_state_1, tf.nn.relu)
        logits = tf.layers.dense(x, config.hidden_state_2)   # [-1, 10]
        # 直接由 logit 获得预测值
        self.y_predict = tf.argmax(logits, axis=1, output_type=tf.int32)
        y_prob = tf.maximum(tf.nn.softmax(logits), 1e-6)
        # y_prob = tf.nn.softmax(logits)   # [-1, 10]

        self.y = tf.placeholder(tf.int32, [None], name='y')  # [-1]
        y = tf.one_hot(self.y, 10)  # [-1, 10]
        ones = tf.reshape(tf.ones([10], tf.float32), [-1, 10])
        # self.loss = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_prob), axis=1))
        # self.loss = -tf.reduce_mean(tf.reduce_sum(y * tf.log(tf.clip_by_value(y_prob, 1e-6, tf.cast(float('inf'), tf.float32))), axis=1))
        self.loss = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_prob) + (ones - y) * tf.log(ones - y_prob), axis=1))
        self.lr = tf.placeholder(tf.float32, (), name='lr')
        opt = tf.train.AdamOptimizer(self.lr)
        self.opt_train = opt.minimize(self.loss)

        precise = tf.cast(tf.equal(self.y, self.y_predict), tf.float32)
        self.precise = tf.reduce_mean(precise)
   

class MNISTApp:
    def __init__(self, config: Config):
        self.session = tf.Session()
        self.config = config
        self.ts = Tensors(config)
        self.data = Samples(config).data
        self.saver = tf.train.Saver()

        try:
            self.saver.restore(self.session, config.save_path)
            print('模型已从 %s 路径下恢复' % config.save_path)
        except:
            self.session.run(tf.global_variables_initializer())
            print('没有保存的模型，模型已进行全局初始化')

    def train(self):
        for epoch in range(self.config.epoches):
            batches = self.data.train.num_examples // self.config.batch_size
            for batch in range(batches):
                xs, ys = self.data.train.next_batch(self.config.batch_size)
                _, loss = \
                        self.session.run([self.ts.opt_train, self.ts.loss], {
                            self.ts.x: xs,
                            self.ts.y: ys,
                            self.ts.lr: self.config.lr
                })
                print('epoch: %d ---- batch: %d ------- loss: %f' % (epoch, batch, loss))
            self.save()
            self.valid()

    def predict(self):
        xs, ys = self.data.test.next_batch(self.config.batch_size)
        y_predict = self.session.run(self.ts.y_predict, {
            self.ts.x: xs
        })
        print('原始值：', xs)
        print('预测值：', y_predict)

    def valid(self):
        xs, ys = self.data.validation.next_batch(self.config.batch_size)
        precise = self.session.run(self.ts.precise, {
            self.ts.x: xs,
            self.ts.y: ys
        })
        print(f'验证集准确率为：{precise*100}%')

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('模型已保存在 %s 路径下' % self.config.save_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()


def main():
    cfg = Config()
    app = MNISTApp(cfg)
    with app:
        app.train()
        app.predict()


if __name__ == '__main__':
    main()