# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p29_tf_sqrt.py
@Description    :  
@CreateTime     :  2020/6/12 10:03
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf

v = tf.get_variable('v', (), tf.float32)
n = tf.placeholder(tf.float32, (), 'n')
loss = tf.square(tf.square(v) - n)
lr = tf.placeholder(tf.float32, (), 'lr')

opt = tf.train.GradientDescentOptimizer(lr)
train_op = opt.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())


def sqrt(n_v, lr_v=0.01, epoches=20000):
    global n, lr
    for _ in range(epoches):
        session.run(train_op, {n: n_v, lr: lr_v})
    return session.run(v)


def main():
    for i in range(1, 21):
        print('sqrt(%s) = %.6f' % (i, sqrt(i)))


if __name__ == '__main__':
    main()