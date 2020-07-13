# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p56_ResNet.py
@Description    :  
@CreateTime     :  2020/7/10 09:21
------------------------------------
@ModifyTime     : 反 Resnet 框架
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


class TransposeResNet:
    def __init__(self, name):
        self.bottleneck = SETTING[name]['bottleneck']
        self.repeats = SETTING[name]['repeats']

    def __call__(self, x, size: int, training: bool, name=None):
        '''
        进行反卷积操作
        :param x: 输入值  [-1, -1]
        :param size: 输出大小，必须为 32 的倍数，默认为224
        :param training:
        :param name:
        :return:
        '''
        height, width = _check(size)

        if name is None:
            global _name_id
            name = 'transpose_resnet_%d' % _name_id
            _name_id += 1

        with tf.variable_scope(name):
            filters = 2048 if self.bottleneck else 512
            # [-1, 2048] 或 [-1, 512]
            x = tf.layers.dense(x, filters, name='fc', activation=tf.nn.relu)
            # [-1, -1, -1, 2048] 或 [-1, -1, -1, 512]
            x = tf.reshape(x, [-1, 1, 1, filters])
            # [-1, 7, 7, 2048] 或 [-1, 7, 7, 512]
            x = tf.layers.conv2d_transpose(x, filters, (height // 32, width // 32), 1, 'same',
                                           name='deconv1', activation=tf.nn.relu)
            # -> [-1, 56, 56, 64]
            x = self._repeat(x, training)

            # 池化对应操作为反卷积
            # x: [-1, 56, 56, 64]  -> [-1, 112, 112, 64]
            x = tf.layers.conv2d_transpose(x, 64, 3, 2, 'same', name='decov2', activation=tf.nn.relu)
            # [-1, 112, 112, 64] -> [-1, 224, 224, 3]
            x = tf.layers.conv2d_transpose(x, 3, (height // 32, width // 32), 2, 'same', name='decov3')
            return x

    def _repeat(self, x, training):
        # [-1, 7, 7, 2048] 或 [-1, 7, 7, 512]  -> [-1, 56, 56, 64]
        filters = x.shape[-1].value
        for num_i, num in zip(range(len(self.repeats) - 1, -1, -1), reversed(self.repeats)):
            for i in range(num-1, -1, -1):
                x = self._transpose_residual(x, num_i, i, filters, training)
            filters //= 2
        return x

    def _transpose_residual(self, x, num_i, i, filters, training):
        strides = 2 if num_i > 0 and i == 0 else 1
        if self.bottleneck:
            left = _my_deconv(x, filters, 1, 1, 'same', name='res_%d_%d_left_myconv1' % (num_i, i),
                            training=training)
            filters //= 4
            left = _my_deconv(left, filters, 3, 1, 'same', name='res_%d_%d_left_myconv2' % (num_i, i),
                            training=training)
            left = _my_deconv(left, filters, 1, strides, 'same', name='res_%d_%d_left_myconv3' % (num_i, i),
                            training=training, active=False)
        else:
            left = _my_deconv(x, filters, 3, 1, 'same', name='res_%d_%d_left_myconv1' % (num_i, i),
                            training=training)
            left = _my_deconv(left, filters, 3, strides, 'same', name='res_%d_%d_left_myconv2' % (num_i, i),
                            training=training)

        if filters != x.shape[-1].value or strides > 1:
            # 如果右侧通道数或图片大小不相等，则通过卷积
            right = _my_deconv(x, filters, 1, strides, 'same', name='res_%d_%d_right_myconv1' % (num_i, i),
                             training=training, active=False)
        else:
            right = x
        return tf.nn.relu(left + right)


def _my_deconv(x, filters, kernel_size, strides, padding, name, training, active: bool=True):
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, filters, kernel_size, strides, padding, name='deconv')
        x = tf.layers.batch_normalization(x, [1, 2, 3], epsilon=1e-6, training=training, name='bn')
        if active:
            x = tf.nn.relu(x)
    return x


def _check(size):
    if type(size) == int:
        size = (size, size)
    height, width = size
    assert height % 32 == 0
    assert width % 32 == 0
    return height, width


def main():
    net = TransposeResNet(RESNET18)
    # 调用 __call__ 函数
    x = net(tf.random_normal([20, 123]), 224, True)  # 使用 （）就可以调用魔法函数__call__'
    print(x.shape)

    net = TransposeResNet(RESNET101)
    x = net(tf.random_normal([20, 123]), 224, True)  # 使用 （）就可以调用魔法函数__call__'
    print(x.shape)


if __name__ == '__main__':
    main()