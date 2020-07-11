# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p29_tf_sqrt_2.py
@Description    :  
@CreateTime     :  2020/6/12 16:41
------------------------------------
@ModifyTime     :  使用梯度下降求救根号2
"""
import tensorflow as tf


v = tf.get_variable('v', (), tf.float32)
n = tf.placeholder(tf.float32, (), name='n')
loss = tf.square(tf.square(v) - n)
lr = tf.placeholder(tf.float32, (), 'lr')
opt = tf.train.GradientDescentOptimizer(lr)
train_op = opt.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())


def sqrt(n_v, lr_v=0.01, epoches=2000):
    global n, lr
    for _ in range(epoches):
        session.run(train_op, {n: n_v, lr: lr_v})
    return session.run(v)


if __name__ == '__main__':
    for i in range(1, 11):
        print('sqrt(%s) = %.6f' % (i, sqrt(i)))
