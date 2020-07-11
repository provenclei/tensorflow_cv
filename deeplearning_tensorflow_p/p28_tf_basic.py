# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  temp.py
@Description    :  
@CreateTime     :  2020/6/11 09:50
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
import numpy as np


def main():
    c = tf.constant(111, dtype=tf.float32, name='a')
    print(c)
    print(c.shape)

    c2 = tf.constant([[1, 2, 3], [2, 3, 4]], name='b')
    # 概念：维度数等价于秩
    print('c2形状：', c2.shape)

    c3 = tf.random_normal([3, 4], dtype=tf.float32)
    # 左闭右开
    c4 = tf.random_uniform([2, 3, 4], 0, 10)
    c5 = tf.random_normal([3, 4, 5])
    c6 = tf.random_normal([2, 4])

    d1 = c + 3
    d2 = c2 + 3

    # 变量
    # 1.always using get_variable
    # 2. the value of a variable can be reserved from a session's run to another one
    # 3. initialize a variable before use it
    v1 = tf.Variable(111, name='v1')
    v2 = tf.get_variable('v2', [3, 4], tf.float32)
    v3 = tf.get_variable('v3', [4, 5], tf.float32)

    # 算数运算：+，-，*，/,//,%,**
    # 关系运算: >,<,<=,>=,==,!=
    # tf.logical_and
    # tf.logical_or
    # tf.logical_not
    # tf.logical_xor
    # tf.cast 强制类型转换

    # tf.concat
    # np.concatenate

    # print(np.random.normal([4, 3]) + np.random.normal([2, 3]))

    # tf.reshape

    v4 = tf.expand_dims(v3, axis=1)
    print('v4', v4.shape)
    #
    # 转置
    v = tf.random_normal([2, 3, 5, 7])
    v5 = tf.transpose(v)
    print('v5', v5.shape)
    v6 = tf.transpose(v, [1, 2, 0, 3])
    print('v6', v6.shape)
    # reshape
    v7 = tf.reshape(v3, [-1, 4])
    print('v7', v7)

    v8 = tf.random_normal([4, 277, 300])
    v11 = tf.random_normal([5])
    # v9 = tf.random_normal([4, 277, 5])

    # v9 = tf.expand_dims(v11, -1)
    v9 = tf.reshape(v11, [-1, -1, tf.Dimension(v11.shape[-1])])
    print('v11', v11.shape)

    v12 = tf.expand_dims(v11, dim=0)
    v12 = tf.expand_dims(v12, dim=1)
    # v12 = tf.reshape(v12, [4, 277, 5])
    print('v12', v12.shape)

    v10 = tf.concat([v8, v12], axis=2)
    print('v10', v10.shape)

    # placeholder 占位符
    p = tf.placeholder(tf.float32, [3, 5, 7], name='p')
    a = p * 3

    with tf.Session() as session:
        c_v = session.run(c)
        print('c:', c_v)
        print('c2:', session.run(c2))
        print('d1:', session.run(d1))
        print('d2:', session.run(d2))
        print('c3:', session.run(c3))
        print('c4:', session.run(c4))
        print('c3 + c4:', session.run(c3 + c4))
        # print('c3 + c5:', session.run(c3 + c5))
        # print('c3 + c6:', session.run(c3 + c6))

        session.run(tf.global_variables_initializer())
        print(session.run(v2))
        print(session.run(tf.matmul(v2, v3)))
        print(session.run(tf.matmul(v2, v3)).shape)

        a_v = session.run(a, {p: np.random.uniform(0, 5, [3, 5, 7])})
        print('a_v', a_v.shape, a_v)


if __name__ == '__main__':
    main()