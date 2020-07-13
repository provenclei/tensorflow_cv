# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p39_VAE_mnist.py
@Description    :  
@CreateTime     :  2020/6/23 10:15
------------------------------------
@ModifyTime     :  手写数字生成

如何计算标准差？
1. 使用惯性系数
2. 将方差公式展开后得：二阶中心距减去均值的平方

"""
import p50_framework as myf
import p59_VAE_facereg as face
import p56_ResNet as resnet
import p58_transpose_ResNet as deresnet

import tensorflow as tf
import numpy as np
import cv2


class MyConfig(face.MyConfig):
    def get_name(self):
        return 'p59'

    def get_sub_tensors(self, gpu_idx):
        return MySubTensors(self)


class MySubTensors(face.MySubTensors):

    def get_loss(self, x):
        loss1 = tf.reduce_mean(tf.abs(self.y - x))
        # x: [-1, img_size, img_size, 3]
        # self.y: [-1, img_size, img_size, 3]
        length = tf.shape(self.y)[0] // 2
        y1 = self.y[: length, :, :, :]
        y2 = self.y[length:, :, :, :]
        x1 = self.x[: length, :, :, :]
        x2 = self.x[length:, :, :, :]

        loss2 = tf.reduce_mean(tf.abs(y1 - y2 - (x1 - x2)))
        loss = loss1 * 2 + loss2
        return loss

    def encode(self, x, vec_size):
        '''
        encode the x to vector which size is vec_size
        :param x: input tensor, shape is [-1, img_size, img_size, 3]
        :param vec_size:
        :return: the semantics vectors which shape is [-1, vec_size]
        '''
        net = resnet.ResNet(resnet.RESNET50)
        # 测试时也会更新
        logits = net(x, vec_size, False, 'resnet')  # logits: [-1, vec_size]
        return logits

    def decode(self, vec):
        '''
        使用反卷积(上采样)，反卷积只能恢复尺寸，不能恢复数值
        the semantics vector
        :param vec: [-1, vec_size]
        :return: [-1, img_size, img_size, 3]
        '''
        net = deresnet.TransposeResNet(deresnet.RESNET50)
        y = net(vec, self.config.img_size, False, 'deresnet')  # [-1, img_size, img_size, 3]
        return y


if __name__ == '__main__':
    cfg = MyConfig()
    cfg.from_cmd()
