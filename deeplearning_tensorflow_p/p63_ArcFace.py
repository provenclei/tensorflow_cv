# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p63_ArcFace.py
@Description    :  
@CreateTime     :  2020/7/16 09:44
------------------------------------
@ModifyTime     :  人脸对比
"""
import tensorflow as tf
from p45_celeba import CelebA
from p48_BufferDS import BufferDS
import p50_framework as myf
import p63_face_recog as p63


class MyConfig(p63.MyConfig):
    def get_name(self):
        return 'p63'

    def __init__(self, persons):
        super(MyConfig, self).__init__(persons)
        self.scale = 30  # gamma

    def get_sub_tensors(self, gpu_idx):
        return MySubTensors(self)

    def test(self):
        app = self.get_app()
        with app:
            pass


class MySubTensors(p63.MySubTensors):
    def get_logits(self, vector):
        # [-1, vec_size]
        vector = tf.nn.l2_normalize(vector, axis=1)
        w = tf.get_variable('std_vector', [vector.shape[-1].value, self.config.persons], tf.float32)
        w = tf.nn.l2_normalize(w, axis=0)
        logits = tf.matmul(vector, w)  # [-1, persons]
        return logits * self.config.scale


def main():
    path_img = '/Users/tenglei/Downloads/face_identity/img/img_ali.gn_cel.eba.zip'
    path_an = '/Users/tenglei/Downloads/face_identity/Anno/identity_CelebA.txt'
    path_bbox = '/Users/tenglei/Downloads/face_identity/Anno/list_bbox_celeba.txt'
    celeba = CelebA(path_img, path_an, path_bbox)
    cfg = MyConfig(celeba.persons)

    ds = BufferDS(cfg.buffer_size, celeba, cfg.batch_size)
    cfg.ds = ds

    cfg.from_cmd()


if __name__ == '__main__':
    main()