# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t56_resnet_framework.py
@Description    :  
@CreateTime     :  2020/7/12 10:21
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf


RESNET18 = 'ResNet18'
RESNET34 = 'ResNet34'
RESNET50 = 'ResNet50'
RESNET101 = 'ResNet101'
RESNET152 = 'ResNet152'

SETTING = {
    RESNET18: {"bottleneck": False, 'repeats': [2, 2, 2, 2]},
    RESNET34: {"bottleneck": False, 'repeats': [3, 4, 6, 3]},
    RESNET50: {"bottleneck": True, 'repeats': [3, 4, 6, 3]},
    RESNET101: {"bottleneck": True, 'repeats': [3, 4, 23, 3]},
    RESNET152: {"bottleneck": True, 'repeats': [3, 8, 36, 3]},
}


_name_id = 1


class ResNet:
    def __init__(self, name):
        self.bottleneck = SETTING[name]['bottleneck']
        self.repeats = SETTING[name]['repeats']

    def __call__(self, x, logits_dim: int, training: bool, name=None):
        height, width = _check(x)
        if name is None:
            global _name_id
            name = 'resnet_%d' % _name_id
            _name_id += 1
        with tf.variable_scope(name):
            x = _my_conv(x, 64, (height // 32, width // 32), 2, 'same', name='conv1', training=False)
            x = tf.layers.max_pooling2d(x, 2, 2, 'same')

            x = self._repeat(x, training)

            x = tf.layers.average_pooling2d(x, (height // 32, width // 32), 1)
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, logits_dim, name='fc')
        return x

    def _repeat(self, x, training):
        filters = 64
        for num_i, num in enumerate(self.repeats):
            for i in range(num):
                x = self._residual(x, num_i, i, filters, training)
            filters *= 2
        return x

    def _residual(self, x, num_i, i, filters, training):
        strides = 2 if num_i > 0 and i == 0 else 1
        if self.bottleneck:
            left = _my_conv(x, filters, 1, strides, 'same', name='res_%d_%d_left_myconv1' % (num_i, i),
                            training=training)
            left = _my_conv(left, filters, 3, 1, 'same', name='res_%d_%d_left_myconv2' % (num_i, i),
                            training=training)
            left = _my_conv(left, 4 * filters, 1, 1, 'same', name='res_%d_%d_left_myconv3' % (num_i, i),
                            training=training, active=False)
        else:
            left = _my_conv(x, filters, 3, strides, 'same', name='res_%d_%d_left_myconv1' % (num_i, i),
                            training=training)
            left = _my_conv(left, filters, 3, 1, 'same', name='res_%d_%d_left_myconv2' % (num_i, i),
                            training=training, active=False)

        if i == 0:
            if self.bottleneck:
                filters *= 4
            right = _my_conv(x, filters, 1, strides, 'same',
                             name='res_%d_%d_right_myconv1' % (num_i, i), training=training, active=False)
        else:
            right = x
        return tf.nn.relu(left + right)


def _check(x):
    shape = x.shape
    assert len(shape) == 4
    height = shape[1].value
    width = shape[2].value
    assert height % 32 == 0
    assert width % 32 == 0
    return height, width


def _my_conv(x, filters, kernel_size, strides, padding, name, training, active: bool = True):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, filters, kernel_size, strides, padding, name='conv')
        x = tf.layers.batch_normalization(x, [1, 2, 3], epsilon=1e-6, training=training, name='bn')
        if active:
            x = tf.nn.relu(x)
        return x


def main():
    net = ResNet(RESNET18)
    x = net(tf.random_normal([20, 32 * 7, 32 * 7, 3]), 100, True)  # 使用 （）就可以调用魔法函数__call__'
    print(x.shape)

    net = ResNet(RESNET101)
    x = net(tf.random_normal([20, 32 * 7, 32 * 7, 3]), 100, True)  # 使用 （）就可以调用魔法函数__call__'
    print(x.shape)


if __name__ == '__main__':
    main()