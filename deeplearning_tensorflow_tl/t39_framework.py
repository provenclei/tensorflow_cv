# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t39_framework.py
@Description    :  
@CreateTime     :  2020/6/23 16:01
------------------------------------
@ModifyTime     :  
"""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import numpy as np
import argparse


def get_gpus():
    value = os.getenv('CUDA_VISIBLE_DEVICES', '0').split(',')
    return len(value)


class Config:
    def __init__(self):
        self.lr = 0.001
        self.epoches = 2000
        self.batch_size = 10
        self.save_path = './models/{name}'.format(name=self.get_name())
        self.sample_path = None
        self.logdir = './log/{name}/{name}'.format(name=self.get_name())
        self.new_model = False
        # GPU 个数
        self.gpus = get_gpus()

    def get_name(self):
        # return 't39'
        raise Exception('方法未被重写！！！')

    def __repr__(self):
        attrs = self.get_attrs()
        result = ['%s = %s' % (key, attrs[key]) for key in attrs]
        return ','.join(result)

    def get_attrs(self):
        result = {}
        for att in dir(self):
            if att.startswith('_'):
                continue
            value = getattr(self, att)
            if value is None or type(value) in (str, int, float, bool):
                result[att] = value
        return result

    def from_cmd(self):
        parser = argparse.ArgumentParser()
        attrs = self.get_attrs()
        for attr in attrs:
            value = attrs[attr]
            if type(value) == bool:
                parser.add_argument('--'+attr, default=value, help='Default is %s' % value,
                                    action='store_%s' % ('false' if value else 'true'))
            else:
                parser.add_argument('--'+attr, default=value, type=type(value),
                                    help='default is %s' % value)
        a = parser.parse_args()
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, getattr(a, attr))

    def get_tensors(self):
        return Tensors()

    def get_app(self):
        return App()


class Tensors:
    def __init__(self):
        with tf.device('/gpu:0'):
            pass


class App:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True
            self.session = tf.Session(graph=graph, config=cfg)

            self.ts = config.get_tensors()
            self.saver = tf.train.Saver()

            if config.new_model:
                self.session.run(tf.global_variables_initializer())
                print('模型完成初始化')
            else:
                try:
                    self.saver.restore(self.session, config.save_path)
                    print('已恢复模型')
                except:
                    print('未发现存储模型，开始初始化模型！')
                    self.session.run(tf.global_variables_initializer())

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('模型保存成功')

    def train(self, ds_train, ds_validation):
        cfg = self.config
        ts = self.ts
        writer = tf.summary.FileWriter(cfg.logdir, )
        batches = ds_train.num_examples // cfg.batch_size

        for epoch in range(cfg.epoches):
            for batch in range(batches):
                _, summary = self.session.run([ts.train_opt, ts.summary], feed_dict=self.get_feed_dict(ds_train))
                writer.add_summary(summary, epoch * batches + batch)
            print('Epoch:', epoch, flush=True)
            self.save()

    def test(self):
        pass

    def get_feed_dict(self, ds):
        values = ds.next_batch(self.config.batch_size)
        return {tensor: value for tensor, value in zip(self.ts.input, values)}


def main():
    # print(get_gpu())
    cfg = Config()
    print(cfg)

    print("-"*30)

    cfg.from_cmd()
    print(cfg)


if __name__ == '__main__':
    main()