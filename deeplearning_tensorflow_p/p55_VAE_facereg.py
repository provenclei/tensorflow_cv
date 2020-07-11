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
# from TF_turial.deeplearning_tensorflow_p import p50_framework as myf
import p50_framework as myf
from p45_celeba import CelebA
from p48_BufferDS import BufferDS

import tensorflow as tf
import numpy as np
import cv2


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.buffer_size = 10
        self.batch_size = 20
        self.epoches = 5
        self.lr = 0.0001/2

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
        return 'p55'

    def get_sub_tensors(self, gpu_idx):
        return MySubTensors(self)

    def get_ds_test(self):
        return None

    def get_ds_train(self):
        if self.ds_train is None:
            self.celeba = CelebA(self.path_img, self.path_an, self.path_bbox)
            self.persons = self.celeba.persons
            self.ds_train = BufferDS(self.buffer_size, self.celeba, self.batch_size)
        return self.ds_train

    def get_app(self):
        # 调用父类的构造函数
        return MyApp(self)


class MySubTensors:
    def __init__(self, config: MyConfig):
        self.config = config
        x = tf.placeholder(tf.float32, [None, None, None, 3], 'x')
        self.inputs = [x]

        # RGB HSV
        x = tf.image.resize_images(x, (config.img_size, config.img_size))/255
        self.vec = self.encode(x, config.vec_size)  # [-1, vec_size]
        self.process_normal(self.vec)  # 注意次序！！！
        # [-1, img_size, img_size, 3]
        self.y = self.decode(self.vec)

        loss = tf.reduce_mean(tf.square(self.y - x))
        self.losses = [loss]

    def process_normal(self, vec):
        '''
        计算平均数和方差(动量法)
        并在计算图中保存变量（assign）
        :param vec: [-1, vec_size]
        :return:
        '''
        mean = tf.reduce_mean(vec, axis=0)  # [vec_size]
        # mean square difference
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
        '''
        encode the x to vector which size is vec_size
        :param x: input tensor, shape is [-1, img_size, img_size, 3]
        :param vec_size:
        :return: the semantics vectors which shape is [-1, vec_size]
        '''
        filters = 32
        # [-1, img_size, img_size, 16]
        x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv1')
        for i in range(5):
            filters *= 2
            # [-1, img_size, img_size, 32]  [-1, 14, 14, 64]
            x = tf.layers.conv2d(x, filters, 3, 2, 'same', activation=tf.nn.relu, name='conv2_%d' % i)
        # [-1, 4, 4, 1024]
        size = self.config.img_size // 2 ** 5  # 4
        x = tf.layers.conv2d(x, vec_size, size, 1, 'valid', name='conv3')  # [-1, 1, 1, vec_sze]
        return tf.reshape(x, [-1, vec_size])

    def decode(self, vec):
        '''
        使用反卷积(上采样)，反卷积只能恢复尺寸，不能恢复数值
        the semantics vector
        :param vec: [-1, vec_size]
        :return: [-1, img_size, img_size, 3]
        '''
        size = self.config.img_size // 2 ** 5  # 4
        filters = 1024
        # [-1 ,vec_size] -> [-1, size*size*filters]
        y = tf.layers.dense(vec, size * size * filters, activation=tf.nn.relu, name='dens_1')
        # [-1, size*size*filters] -> [-1, size, size, filters]
        y = tf.reshape(y, [-1, size, size, filters])
        for i in range(5):
            filters //= 2
            # 两次反卷积 ：[-1, 14， 14, 32]  [-1, 28, 28, 16]
            y = tf.layers.conv2d_transpose(y, filters, 3, 2, 'same', activation=tf.nn.relu, name='deconv1_%d'%i)
        # [-1, img_size, img_size, 32]
        y = tf.layers.conv2d_transpose(y, 3, 3, 1, 'same', name='deconv2')  # [-1, img_size, img_size, 3]
        return y


class MyDS:
    def __init__(self, ds, config):
        self.ds = ds
        self.lr = config.lr
        self.num_examples = ds.num_examples

    def next_batch(self, batch_size):
        xs, ys = self.ds.next_batch(batch_size)
        return xs


class MyApp(myf.App):
    def after_batch(self, epoch, batch):
        if batch % 500 == 0:
            print('epoch: %d ------- batch: %d' % (epoch, batch))

    def test(self, ds_test):
        cfg = self.config
        mean = self.session.run(self.ts.sub_ts[0].final_mean)
        print(mean)
        msd = self.session.run(self.ts.sub_ts[0].final_msd)  # 二阶原点矩
        std = np.sqrt(msd - mean ** 2)
        print(std)
    
        vec = np.random.normal(mean, std, [cfg.batch_size, len(std)])  # [-1, 4]
        imgs = self.session.run(self.ts.sub_ts[0].y, {self.ts.sub_ts[0].vec: vec})  # [-1, img_size, img_size]

        imgs = np.reshape(imgs, [-1, cfg.cols, cfg.img_size, cfg.img_size, 3])  # [-1, 20, 28, 28]
        imgs = np.transpose(imgs, [0, 2, 1, 3, 4])  # [-1, img_size, 5, img_size]
        imgs = np.reshape(imgs, [-1, cfg.cols*cfg.img_size, 3])  # [-1, 20*img_size]
        cv2.imwrite(cfg.img_path, imgs*255)


if __name__ == '__main__':
    cfg = MyConfig()
    cfg.from_cmd()
