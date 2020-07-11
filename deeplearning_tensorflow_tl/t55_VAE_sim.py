# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t55_VAE_sim.py
@Description    :  
@CreateTime     :  2020/7/9 15:45
------------------------------------
@ModifyTime     :  相似人脸的过渡效果
"""
import t55_framework as myf
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
        self.lr = 0.0001 / 2

        # 模型参数
        self.path_img = '/Users/tenglei/Downloads/face_identity/img/img_ali.gn_cel.eba.zip'
        self.path_an = '/Users/tenglei/Downloads/face_identity/Anno/identity_CelebA.txt'
        self.path_bbox = '/Users/tenglei/Downloads/face_identity/Anno/list_bbox_celeba.txt'

        self.celeba = None
        self.ds_train = None

        self.img_size = 32 * 4  # target size of images
        self.vec_size = 100
        # 惯性系数
        self.momentum = 0.99

        # 打印参数
        self.cols = 4
        self.img_path = './imgs/{name}/test.jpg'.format(name=self.get_name())

    def get_name(self):
        return 't55'

    def get_sub_tensors(self, gpu_idx):
        return MySubTensors(self)

    def get_ds_test(self):
        return None

    def get_ds_train(self):
        if self.ds_train is None:
            self.celeba = CelebA(self.path_img, self.path_an)
            self.persons = self.celeba.persons
            self.ds_train = BufferDS(self.buffer_size, self.celeba, self.batch_size)
        return self.ds_train

    def get_app(self):
        # 调用父类的构造函数
        return MyApp(self)

    def test(self):
        with self.get_app() as app:
            app.transfer()


class MySubTensors:
    '''
    inpus, losses
    '''
    def __init__(self, config: MyConfig):
        self.config = config
        x = tf.placeholder(tf.float32, [None, None, None, 3], 'x')
        self.inputs = [x]

        x = tf.image.resize_images(x, (config.img_size, config.img_size)) / 255
        self.vec = self.encode(x, config.vec_size)
        self.process_normal(self.vec)
        self.y = self.decode(self.vec)

        loss = tf.reduce_mean(tf.square(self.y - x))
        self.losses = [loss]

    def process_normal(self, vec):
        mean = tf.reduce_mean(vec, axis=0)
        msd = tf.reduce_mean(tf.square(vec), axis=0)

        vector_size = vec.shape[1].value
        self.final_mean = tf.get_variable('mean', [vector_size], tf.float32, tf.initializers.zeros, trainable=False)
        self.final_msd = tf.get_variable('msd', [vector_size], tf.float32, tf.initializers.zeros, trainable=False)

        mom = self.config.momentum
        assign = tf.assign(self.final_mean, self.final_mean * mom + mean * (1 - mom))
        # 建立 assign 与 train_op 的控制依赖，正向传播是走实线和虚线，反向传播不去求控制依赖的值
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign)

        assign = tf.assign(self.final_msd, self.final_msd * mom + msd * (1 - mom))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign)

    def encode(self, x, vec_size):
        filters = 32
        x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv1')
        for i in range(5):
            filters *= 2
            x = tf.layers.conv2d(x, filters, 3, 2, 'same', activation=tf.nn.relu, name='conv2_%d' % i)
        size = self.config.img_size // 2 ** 5
        x = tf.layers.conv2d(x, vec_size, size, 1, 'valid', name='conv3')  # [-1, 1, 1, vec_size]
        return tf.reshape(x, [-1, vec_size])

    def decode(self, vec):
        size = self.config.img_size // 2 ** 5
        filters = 1024
        y = tf.layers.dense(vec, size * size * filters, activation=tf.nn.relu, name='dens_1')
        y = tf.reshape(y, [-1, size, size, filters])
        for i in range(5):
            filters //= 2
            y = tf.layers.conv2d_transpose(y, filters, 3, 2, 'same', activation=tf.nn.relu, name='deconv1_%d'%i)
        y = tf.layers.conv2d_transpose(y, 3, 3, 1, 'same', name='deconv2')
        return y


class MyApp(myf.App):
    def after_batch(self, epoch, batch):
        if batch % 500 == 0:
            print('epoch: %d ------- batch: %d' % (epoch + 1, batch))

    def test(self, ds_test):
        pass

    def transfer(self):
        path = ['./faces/1.jpg', './faces/0.jpg']
        size = self.config.img_size
        imgs = [cv2.resize(cv2.imread(p), (size, size)) for p in path]
        ts = self.ts.sub_ts[0]
        vecs = self.session.run(ts.vec, {ts.inputs[0]: imgs})
        vecs = get_middle_vectors(vecs[0], vecs[1])
        pics = self.session.run(ts.y, {ts.vec: vecs}) * 255
        pics = [imgs[1]] + list(pics) + [imgs[0]]
        pics = np.reshape(pics, [-1, self.config.img_size, 3])

        myf.make_dirs(self.config.img_path)
        cv2.imwrite(self.config.img_path, pics)
        print('write image into', self.config.img_path)


def get_middle_vectors(src, dst):
    num = 10
    delta = 1 / (num + 1)
    alpha = delta  # 系数
    result = []
    for _ in range(num):
        vec = src * alpha + dst * (1 - alpha)
        result.append(vec)
        alpha += delta
    return result


def main():
    cfg = MyConfig()
    cfg.from_cmd()


if __name__ == '__main__':
    main()