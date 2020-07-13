# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t57_my_bn.py
@Description    :  
@CreateTime     :  2020/7/12 14:55
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf


_name_id = 1


def my_bn(x, axis, training=True, momentum=0.99, epslon=1e-6, name=None):
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

        mean = tf.reduce_mean(x, compute_axis)
        msd = tf.reduce_mean(tf.square(x), compute_axis)

        shape = [x.shape[e].value if e in axis else 1 for e in range(4)]
        mean = tf.reshape(mean, shape)
        msd = tf.reshape(msd, shape)

        final_mean = tf.get_variable('fm', shape, tf.float32, trainable=False)
        final_msd = tf.get_variable('fmsd', shape, tf.float32, trainable=False)

        assign_mean = tf.assign(final_mean, momentum * final_mean + (1 - momentum) * mean)
        assign_msd = tf.assign(final_msd, momentum * final_msd + (1 - momentum) * msd)

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign_msd)

        std = tf.maximum(final_msd - tf.square(final_mean), epslon)
        assign_std = tf.maximum(assign_msd - tf.square(assign_mean), epslon)
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