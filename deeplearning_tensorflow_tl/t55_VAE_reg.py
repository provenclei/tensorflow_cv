# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t55_VAE_reg.py
@Description    :  
@CreateTime     :  2020/7/7 17:04
------------------------------------
@ModifyTime     :  
"""
# from TF_turial.deeplearning_tensorflow_tl import t50_framework as myf
import t50_framework as myf
from t45_celeba import CelebA
from t48_BufferDS import BufferDS

import tensorflow as tf
import numpy as np
import cv2


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.buffer_size = 10
        self.batch_size = 20
        self.epoches = 5
        self.lr = 0.0001

        # 模型参数
        self.path_img = '/Users/tenglei/Downloads/face_identity/img/img_ali.gn_cel.eba.zip'
        self.path_an = '/Users/tenglei/Downloads/face_identity/Anno/identity_CelebA.txt'
        self.path_bbox = '/Users/tenglei/Downloads/face_identity/Anno/list_bbox_celeba.txt'

        # self.celeba = None
        self.ds_train = None

        self.img_size = 32 * 4  # target size of images
        self.vec_size = 100
        # 惯性系数
        self.momentum = 0.99

        # 打印参数
        self.cols = 4
        self.img_path = './imgs/{name}/test.jpg'.format(name=self.get_name())

    def get_app(self):
        return MyApp(self)

    def get_name(self):
        return 't55'

    def get_sub_tensors(self, gpu_idx):
        return MySubTensors(self)

    def get_ds_train(self):
        if self.ds_train is None:
            self.celeba = CelebA(self.path_img, self.path_an)
            self.persons = self.celeba.persons
            self.ds_train = BufferDS(self.buffer_size, self.celeba, self.batch_size)
        return self.ds_train

    def get_ds_test(self):
        return None


class MySubTensors:
    def __init__(self, config: MyConfig):
        self.config = config
        x = tf.placeholder(tf.float32, [None, None, None, 3], name='x')
        self.inputs = [x]

        x = tf.image.resize_images(x, (config.img_size, config.img_size))/255
        self.vec = self.encode(x, config.vec_size)
        self.process_normal(self.vec)
        self.y = self.decode(self.vec)

        loss = tf.reduce_mean(tf.square(self.y - x))
        self.losses = [loss]

    def encode(self, x, vec_size):
        filters = 32
        # [-1, img_size, img_size, 16]
        x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv1')
        for i in range(5):
            filters *= 2
            # [-1, img_size, img_size, 32]  [-1, 14, 14, 64]
            x = tf.layers.conv2d(x, filters, 3, 2, 'same', activation=tf.nn.relu, name='conv2_%d' % i)
        # [-1, 4, 4, 1024]
        size = self.config.img_size // 2 ** 5  # 4
        x = tf.layers.conv2d(x, vec_size, size, 1, 'valid', name='conv3')  # [-1, 1, 1, vec_size]
        return tf.reshape(x, [-1, vec_size])

    def decode(self, vec):
        size = self.config.img_size // 2 ** 5  # 4
        filters = 1024
        # [-1 ,vec_size] -> [-1, size*size*filters]
        y = tf.layers.dense(vec, size * size * filters, activation=tf.nn.relu, name='dens_1')
        # [-1, size*size*filters] -> [-1, size, size, filters]
        y = tf.reshape(y, [-1, size, size, filters])
        for i in range(5):
            filters //= 2
            # 两次反卷积 ：[-1, 14， 14, 32]  [-1, 28, 28, 16]
            y = tf.layers.conv2d_transpose(y, filters, 3, 2, 'same', activation=tf.nn.relu, name='deconv1_%d' % i)
        # [-1, img_size, img_size, 32]
        y = tf.layers.conv2d_transpose(y, 3, 3, 1, 'same', name='deconv2')  # [-1, img_size, img_size, 3]
        return y

    def process_normal(self, vec):
        pass


class MyDS:
    def __init__(self, ds, config: MyConfig):
        self.ds = ds
        self.lr = config.lr
        self.num_examples = ds.num_examples

    def next_batch(self, batch_size):
        xs, _ = ds.next_batch(batch_size)
        return xs, self.lr


class MyApp(myf.App):
    def test(self):
        pass

    def after_batch(self, epoch, batch):
        if batch % 10 == 0:
            print('epoche: %d------batch: %d' % (epoch, batch))


def main():
    cfg = MyConfig()
    # cfg.from_cmd()
    cfg.test()


if __name__ == '__main__':
    main()