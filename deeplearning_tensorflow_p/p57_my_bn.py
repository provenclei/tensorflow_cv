# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p57_my_bn.py
@Description    :  
@CreateTime     :  2020/7/10 11:19
------------------------------------
@ModifyTime     :  实现 batch_normalization 操作
x = tf.layers.conv2d(x, 64, (height // 32, width // 32), 2, 'same', name='conv1')
inputs,
            axis=-1,
            momentum=0.99,
            epsilon=1e-3,    标准差小于该值，则使用该值

            center=True,     平均值相关参数，True 表示使用 bata 当做可训练参数进行训练
            scale=True,      标准差相关参数，True 表示使用 gamma 当做可训练参数进行训练
            beta_initializer=init_ops.zeros_initializer(),   # 默认初始化值
            gamma_initializer=init_ops.ones_initializer(),   # 默认初始化值

            moving_mean_initializer=init_ops.zeros_initializer(),   # 移动平均数初始值
            moving_variance_initializer=init_ops.ones_initializer(),  # 均方差初始值

            将每个数进行标准化，若遇见某个数不需要标准化时，使用该参数进行挽回
            使得 y*bata + gamma 其中 bata，gamma 都是可训练的！！！对该值在某种程度上进行恢复

            beta_regularizer=None,
            gamma_regularizer=None,

            beta_constraint=None,
            gamma_constraint=None,

            training=False,   # 训练模式调用还是测试中调用

            trainable=True,   # 是否可训练，训练过程中更新动量，预测过程中动量值不发生改变

            name=None,
            reuse=None,   可训练变量是否重用，即多个GPU中共享一个，还是各有一个
            renorm=False,  是否再次进行归一化
            renorm_clipping=None,
            renorm_momentum=0.99,

"""
import tensorflow as tf


_name_id = 1


def my_bn(x, axis, training=True, momentum=0.99, epslon=1e-6, name=None):
    '''
    batch normalization
    :param x:
    :param axis:
    :param training: 张量
    :param momentum:
    :param epslon:
    :param name:
    :return:
    '''
    assert len(x.shape) == 4

    if type(training) == bool:
        training = tf.constant(training)

    if name is None:
        global _name_id
        name = 'my_bn_%d' % _name_id
        _name_id += 1

    with tf.variable_scope(name):
        if type(axis) == int:
            axis = [axis]
        else:
            assert type(axis) == list
        compute_axis = [e for e in range(4) if e not in axis]

        # [224, 224]
        mean = tf.reduce_mean(x, compute_axis)
        msd = tf.reduce_mean(tf.square(x), compute_axis)

        # [1, 224, 224]
        shape = [x.shape[e].value if e in axis else 1 for e in range(4)]
        # [1, 224, 224, 1]
        mean = tf.reshape(mean, shape)
        # [1, 224, 224, 1]
        msd = tf.reshape(msd, shape)

        # final_mean = tf.get_variable(name='fm', shape=shape, dtype=tf.float64, trainable=False)
        # final_msd = tf.get_variable(name='fmsd', shape=shape, dtype=tf.float64, trainable=False)
        final_mean = tf.get_variable('fm', shape, tf.float32, trainable=False)
        final_msd = tf.get_variable('fmsd', shape, tf.float32, trainable=False)

        assign_mean = tf.assign(final_mean, momentum * final_mean + (1 - momentum) * mean)
        assign_msd = tf.assign(final_msd, momentum * final_msd + (1 - momentum) * msd)

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign_msd)

        std = tf.maximum(final_msd - tf.square(final_mean), epslon)
        assign_std = tf.maximum(assign_msd - tf.square(assign_mean), epslon)
        # 问题：这一轮的变量 final_mean 和 final_msd 是上一轮，应该使用 assign_mean 和 assign_std
        # 但是 assign_mean 和 assign_std 是assign，无法获取，所以使用 cond 函数
        # `true_fn` and `false_fn` both return lists of output tensors.
        # `true_fn` and `false_fn` must have the same non-zero number and type of outputs.
        x = tf.cond(training, lambda: (x - assign_mean) / assign_std, lambda: (x - final_mean) / std)
        return x


def main():
    training = tf.placeholder(tf.bool, [], 'training')
    x = tf.random_normal([20, 224, 224, 3])
    y = my_bn(x, [1, 2], training=training)
    print(y.shape)

    y = my_bn(x, [1, 2], training=False)
    print(y.shape)


if __name__ == '__main__':
    main()