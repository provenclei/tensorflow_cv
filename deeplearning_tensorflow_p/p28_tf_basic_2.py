# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p28_tf_basic_2.py
@Description    :  
@CreateTime     :  2020/6/14 20:39
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf


def main():
    a = tf.placeholder(tf.float32, [None, 1])
    b = tf.placeholder(tf.float32, [None, 277, 300])
    c = [a] * 277   # [277, ?, 1]
    print(a.shape, b.shape)
    c = tf.transpose(c, [1, 0, 2])  # [?, 277, 1]
    c = tf.concat((c, b), axis=2)  # [?, 277, 301]
    print(c.shape)


if __name__ == '__main__':
    main()