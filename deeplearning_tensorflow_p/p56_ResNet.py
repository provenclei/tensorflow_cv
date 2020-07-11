# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p56_ResNet.py
@Description    :  
@CreateTime     :  2020/7/10 09:21
------------------------------------
@ModifyTime     : Resnet 框架
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

    def __call__(self, x, logits: int, training: bool, name=None):
        # 检查形状，因为图片缩小了32倍，所以输入形状必须是32的倍数
        height, width = _check(x)
        if name is None:
            global _name_id
            name = 'resnet_%d' % _name_id
            _name_id += 1
        with tf.variable_scope(name):
            # [-1, h/2, w/2, 64]
            # x = tf.layers.conv2d(x, 64, (height // 32, width // 32), 2, 'same', name='conv1')
            # x = tf.layers.batch_normalization(x, axis=(1, 2, 3), epsilon=1e-6, training=training)
            # x = tf.nn.relu(x)
            x = _my_conv(x, 64, (height // 32, width // 32),  2, 'same', name='conv1', training=False)
            # [-1, h/4, w/4, 64]
            x = tf.layers.max_pooling2d(x, 2, 2, 'same')
            x = self._repeat(x, training)
            # [-1, 1, 1, 2048]
            x = tf.layers.average_pooling2d(x, (height // 32, width // 32), 1)
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, logits, name='fc')
            return x

    def _repeat(self, x, training):
        # 起始通道数
        filters = 64
        for num_i, num in enumerate(self.repeats):
            for i in range(num):
                x = self._residual(x, num_i, i, filters, training)
            filters *= 2
        return x

    def _residual(self, x, num_i, i, filters, training):
        '''
        1. 第一个模块中第一层步长为1，其他模块的第一层步长为2，为使图片尺寸缩小2倍
        2. 非瓶颈层，经过两个 3*3 卷积层，不需激活
        3. 瓶颈层的模块，每个模块中瓶颈层最后一层，要使通道数扩大2倍
        3. 没有瓶颈层的模块，直接线性映射
        :param x:
        :param num_i:
        :param i:
        :param filters:
        :param training:
        :return:
        '''
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
            right = _my_conv(x, filters, 1, strides, 'same', name='res_%d_%d_right_myconv1' % (num_i, i),
                             training=False, active=False)
        else:
            right = x
        return tf.nn.relu(left + right)


def _my_conv(x, filters, kernel_size, strides, padding, name, training, active: bool=True):
    '''
    卷积 + NB + Relu(可选择，默认有Relu)
    :param x:
    :param filters:
    :param kernal_size:
    :param strides:
    :param padding:
    :param name:
    :param training: 是否在训练，类型为 bool tensor
    :param active: 激活状态有 Relu 激活
    :return:
    '''
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, filters, kernel_size, strides, padding, name='conv')
        x = tf.layers.batch_normalization(x, [1, 2, 3], epsilon=1e-6, training=training, name='bn')
        if active:
            x = tf.nn.relu(x)
    return x


def _check(x):
    # [-1, h, w, c]
    shape = x.shape
    assert len(shape) == 4
    # 如果为空None, 则报错,所以取value
    height = shape[1].value
    assert height % 32 == 0
    width = shape[2].value
    assert width % 32 == 0
    return height, width


def main():
    net = ResNet(RESNET18)
    # 调用 __call__ 函数
    x = net(tf.random_normal([20, 32 * 7, 32 * 7, 3]), 100, True)  # 使用 （）就可以调用魔法函数__call__'
    print(x.shape)

    net = ResNet(RESNET101)
    x = net(tf.random_normal([20, 32 * 7, 32 * 7, 3]), 100, True)  # 使用 （）就可以调用魔法函数__call__'
    print(x.shape)


if __name__ == '__main__':
    main()