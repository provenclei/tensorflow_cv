# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t39_framework_3.py
@Description    :  
@CreateTime     :  2020/6/28 18:15
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import os
import argparse


def get_gpus():
    value = os.getenv('CUDA_VISIAL_DEVICES', '0').split(',')
    return len(value)


class Config:
    def __init__(self):
        self.lr = 0.001
        self.epoches = 50
        self.batch_size = 10
        self.save_path = './models/{name}/{name}'.format(name=self.get_name())
        self.sample_path = None
        self.logdir = './logs/{name}/'.format(name=self.get_name())
        self.new_model = False
        # GPU 个数
        self.gpus = get_gpus()

    def get_name(self):
        raise Exception('方法未重写!!!')

    def getatts(self):
        result = {}
        for attrs in dir(self):
            if attrs.startswith('_'):
                continue
            value = getattr(self, attrs)
            if value is None or type(value) in (int, float, bool, str):
                result[attrs] = value
        return result

    def cmd_param(self):
        parser = argparse.ArgumentParser()
        attrs = self.getatts()
        for attr in attrs:
            value = attrs[attr]
            if type(value) is bool:
                parser.add_argument('--'+attr, default=value, help='default is %s' % value,
                                    action='store_%s' % ('false' if value else 'true'))
            else:
                parser.add_argument('--'+attr, default=value, type=type(value),
                                    help='default is %s' % value)
        a = parser.parse_args()
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, getattr(a, attr))

    def gettensors(self):
        return Tensors()

    def getapp(self):
        return App(self)

    def __repr__(self):
        attrs = self.getatts()
        result = ['%s = %s' % (key, attrs[key]) for key in attrs]
        return ','.join(result)


class Tensors:
    def __init__(self):
        with tf.device('/gpu:0'):
            pass


class App:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            self.ts = config.gettensors()
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True
            self.session = tf.Session(graph=graph, config=cfg)
            self.saver = tf.train.Saver()
            if config.new_model:
                print('不加载新模型，正在初始化。。。')
                self.session.run(tf.global_variables_initializer())
            else:
                try:
                    self.saver.restore(self.session, config.save_path)
                    print('正在恢复模型')
                except:
                    print('没有要恢复的模型，正在初始化。。。')
                    self.session.run(tf.global_variables_initializer())

    def train(self, ds_train, ds_validation):
        writer = tf.summary.FileWriter(self.config.logdir, graph=self.session.graph)
        batches = ds_train.num_examples // self.config.batch_size
        for epoch in range(self.config.epoches):
            for batch in range(batches):
                _, summary = self.session.run([self.ts.train_op, self.ts.summary], feed_dict=self.get_feeddict(ds_train))
                writer.add_summary(summary, global_step=epoch*batches+batch)
            print('Epoch:', epoch, flush=True)
            self.save()
        writer.close()

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('模型已保存！')

    def get_feeddict(self, ds):
        values = ds.next_batch(self.config.batch_size)
        return {key: value for key, value in zip(self.ts.input, values)}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def main():
    config = Config()
    print(config)
    print('-'*30)
    config.cmd_param()
    print(config)

    dss = read_data_sets(config.sample_path)
    app = App(config)
    with app:
        app.train()


if __name__ == '__main__':
    main()