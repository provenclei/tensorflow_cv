# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t29_tf_basic.py
@Description    :  
@CreateTime     :  2020/6/12 14:35
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
import numpy as np


def main():
    # 常数
    c1 = tf.constant(123, dtype=tf.float32, name='c')
    print('c1:', c1, c1.shape)  # ()表示零维张量，即标量

    # 随机初始化
    c2 = tf.random_normal([2, 3], 0, 1, tf.float32, name='c2', seed=12345)
    c3 = tf.random_uniform([2, 3], 0, 1, tf.float32, name='c3')
    print('c2:', c2, c2.shape)
    print('c3:', c3, c3.shape)

    # 变量
    v1 = tf.Variable(212., name='v1')
    v2 = tf.get_variable('v2', [2, 3], dtype=tf.float32)
    v3 = tf.get_variable('v3', [3, 1], dtype=tf.float32)

    # 转置，reshape，增加维度
    v4 = tf.get_variable('v4', [4, 2, 3], tf.float32)
    v5 = tf.expand_dims(v4, axis=1)
    print('v5:', v5.shape)  # [4, 1, 2, 3]
    v6 = tf.random_normal([2, 3, 4])
    v7 = tf.transpose(v6, [2, 0, 1])
    print('v7:', v7.shape)
    v8 = tf.transpose(v6)
    print('v8:', v8.shape)
    v9 = tf.reshape(v8, [2, 2, -1])
    print('v9:', v9.shape)

    v10 = tf.get_variable('v10', [2, 1, 3], tf.float32)
    v11 = tf.get_variable('v11', [1, 3], tf.float32)
    v12 = tf.get_variable('v12', [3, 3], tf.float32)
    v13 = tf.get_variable('v13', [2, 3, 3], tf.float32)

    # placeholder
    p1 = tf.placeholder(tf.float32, [None], name='p1')
    p2 = tf.placeholder(tf.float32, [2, 2], name='p2')
    a = p2 * 2

    # 创建会话
    with tf.Session() as session:
        print('c1:', session.run(c1))
        print('c2:', session.run(c2))
        print('c3:', session.run(c3))

        # 需要统一进行初始化
        session.run(tf.global_variables_initializer())
        print('v2:', session.run(v1))
        print('v3:', session.run(v2))
        # 矩阵相乘
        print('v1.v2:', session.run(tf.matmul(v2, v3)).shape)

        x = np.array([1, 2, 3, 4])
        print(session.run(p1, {p1: x}))
        print(session.run(a, {p2: x.reshape([2, 2])}))

        # 广播机制
        # v2: [2, 3]
        # v3:[3, 1]
        # v10:[2, 1, 3]
        # v11:[1, 3]
        # v12:[3, 3]
        # v13:[2, 3, 3]
        print('广播v2 + v11：', (v2 + v11).shape)  # [2, 3] + [1, 3] = [2, 3]
        # print('广播v2 + v12：', v2 + v12)  # [2, 3] + [3, 3]无法广播
        print('广播v3 + 1：', (v3 + 1).shape)  # [3, 1] + ()
        print('广播v2 + v10：', (v2 + v10).shape)  # [2, 3] + [2, 1, 3] = [2, 2, 3]
        # print('广播v2 + v13：', (v2 + v13).shape)  # [2, 3] + [2, 3, 3] 无法广播


if __name__ == '__main__':
    main()