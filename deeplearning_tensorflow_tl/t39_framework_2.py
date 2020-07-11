# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t39_framework_2.py
@Description    :  
@CreateTime     :  2020/6/25 22:59
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import os
import argparse


def get_gpus():
    value = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    value = value.split(',')
    return len(value)


def make_dir(path: str):
    pos = path.rfind(os.sep)
    if pos < 0:
        raise Exception('未找到，需提供路径!!!')
    path = path[:pos]
    os.makedirs(path, exist_ok=True)

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
        raise Exception('方法未重写！！！')

    def get_attrs(self):
        result = {}
        for attr in dir(self):
            if attr.startswith('_'):
                continue
            value = getattr(self, attr)
            if value is None or type(value) in (float, int, str, bool):
                result[attr] = value
        return result

    def cmd_parmerter(self):
        parser = argparse.ArgumentParser()
        attrs = self.get_attrs()
        for attr in attrs:
            value = attrs[attr]
            if type(value) == bool:
                parser.add_argument('--'+attr, default=value, help='default value is %s' % value,
                                    action='store_%s' % ('false' if value else 'true'))
            else:
                parser.add_argument('--'+attr, type=type(value), default=value,
                                    help='default value is %s' % value)
        a = parser.parse_args()
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, getattr(a, attr))

    def get_tensors(self):
        return Tensors()

    def get_app(self):
        return App(self)

    def __repr__(self):
        attrs = self.get_attrs()
        result = ['%s = %s' % (attr, attrs[attr]) for attr in attrs]
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
            self.ts = config.get_tensors()
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True
            self.session = tf.Session(config=cfg, graph=graph)
            self.saver = tf.train.Saver()
            if config.new_model:
                self.session.run(tf.global_variables_initializer())
                print('使用了新模型，已进行初始化！')
            else:
                try:
                    self.saver.restore(self.session, config.save_path)
                    print('已加载新模型')
                except:
                    self.session.run(tf.global_variables_initializer())
                    print('没有存储的模型，将重新进行初始化！')

    def train(self, ds_train, ds_validation):
        writer = tf.summary.FileWriter(self.config.logdir, graph=self.session.graph)
        batches = ds_train.num_examples // self.config.batch_size

        for epoch in range(self.config.epoches):
            for batch in range(batches):
                _, summary = self.session.run([self.ts.train_op, self.ts.summary], feed_dict=self.get_feed_dict(ds_train))
                writer.add_summary(summary, epoch * batches + batch)
            print('Epoch:', epoch, flush=True)
            self.save()
        writer.close()

    def get_feed_dict(self, ds):
        values = ds.next_batch(self.config.batch_size)
        return {tensor: value for tensor, value in zip(self.ts.input, values)}

    def test(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    def close(self):
        self.session.close()

    def save(self):
        self.saver.save(self.session, self.config.save_path)


def main():
    config = Config()
    print(config)
    print('-'*50)
    config.cmd_parmerter()
    print(config)

    dss = read_data_sets(config.sample_path)
    app = config.get_app()
    with app:
        app.train(dss.train, dss.validation)


if __name__ == '__main__':
    main()