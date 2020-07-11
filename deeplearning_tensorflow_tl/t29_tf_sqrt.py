# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t29_tf_sqrt.py
@Description    :  
@CreateTime     :  2020/6/12 17:45
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf


v = tf.get_variable('v', (), tf.float32)
n = tf.placeholder(tf.float32, (), name='n')
loss = tf.square(tf.square(v) - n)
lr = tf.placeholder(tf.float32, (), name='lr')
opt = tf.train.AdamOptimizer(lr)
train_op = opt.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())


def sqrt(n_v, lr_v=0.05, epoches=2000):
    global n, lr
    for _ in range(epoches):
        session.run(train_op, {n: n_v, lr: lr_v})
    return session.run(v)


def main():
    for i in range(1, 11):
        print('sqrt(%s) = %s' % (i, sqrt(i)))


if __name__ == '__main__':
    main()