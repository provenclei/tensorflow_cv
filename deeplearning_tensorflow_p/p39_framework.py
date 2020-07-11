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
"""
import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import os


def get_gpus():
    '''
    getenv 函数：获取环境变量
    第一个参数：环境变量名
    第二个参数：默认环境变量值，实际上不会给不存在的环境变量赋值，也就是给一个如果该环境变量不存在时的返回值（可选）
    返回值：字符串
    :return:
    '''
    # 获取环境变量
    value = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    value = value.split(',')
    return len(value)


def make_dirs(path: str):
    pos = path.rfind(os.sep)
    if pos < 0:
        raise Exception('未找到，需提供路径!!!')
    path = path[: pos]
    # 路径存在不需创建
    os.makedirs(path, exist_ok=True)


class Config:
    def __init__(self):
        self.lr = 0.001
        self.epoches = 50
        self.batch_size = 10
        self.save_path = './models/{name}/{name}'.format(name=self.get_name())
        self.sample_path = None
        self.logdir = './logs/{name}'.format(name=self.get_name())
        self.new_model = False
        # GPU 个数
        self.gpus = get_gpus()

    def get_name(self):
        raise Exception('get_name 没有重定义')

    def __repr__(self):
        '''
        将获取的属性按照格式化进行打印
        :return:
        '''
        attrs = self.get_attrs()
        result = ['%s = %s' % (key, attrs[key]) for key in attrs]
        return ','.join(result)

    def get_attrs(self):
        '''
        获取每个对象自身的属性值，并以字典的方式返回
        :return:
        '''
        result = {}
        for attr in dir(self):
            if attr.startswith('_'):
                continue
            value = getattr(self, attr)
            if value is None or type(value) in (int, float, bool, str):
                result[attr] = value
        return result

    def from_cmd(self):
        '''
        从命令行中获取并更改参数
        :return:
        '''
        parser = argparse.ArgumentParser()
        attrs = self.get_attrs()
        # 将每个属性和对应的值加到 parser 中
        # 根据已有属性的类型判断保存方式
        for attr in attrs:
            value = attrs[attr]
            t = type(value)
            if t == bool:
                parser.add_argument('--'+attr, default=value, help='default is %s' % value,
                                    action='store_%s' % ('false' if value else 'true'))
            else:
                parser.add_argument('--'+attr, type=t, default=value, help='default is %s' % value)
        a = parser.parse_args()
        # 将 parser 中的属性解析后添加到对象中
        for attr in attrs:
            if hasattr(self, attr):
                # 将 a 中的参数设置到实例中
                setattr(self, attr, getattr(a, attr))

    def get_tensors(self):
        return Tensors()

    def get_app(self):
        return App(self)


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
            # 没有GPU的时候使用CPU模拟GPU
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True
            self.session = tf.Session(config=cfg, graph=graph)
            self.saver = tf.train.Saver()
            if config.new_model:
                self.session.run(tf.global_variables_initializer())
                print('模型初始化完成')
            else:
                try:
                    self.saver.restore(self.session, config.save_path)
                    print('模型恢复成功')
                except:
                    print('模型恢复失败，初始化中~')
                    self.session.run(tf.global_variables_initializer())

    def close(self):
        self.session.close()

    def train(self, ds_train, ds_validation):
        cfg = self.config
        ts = self.ts
        writer = tf.summary.FileWriter(cfg.logdir, self.session.graph)
        batches = ds_train.num_examples // cfg.batch_size

        for epoch in range(cfg.epoches):
            for batch in range(batches):
                _, summary = self.session.run([ts.train_op, ts.summary], self.get_feed_dict(ds_train))
                writer.add_summary(summary, epoch * batches + batch)
            print('Epoch:', epoch, flush=True)
            self.save()

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('模型保存成功')

    def predict(self, ds):
        pass

    def get_feed_dict(self, ds):
        # 张量都在 inputs 里面，values 中有 xs, ys
        # next_batch 是 ds_train 的对象方法
        values = ds.next_batch(self.config.batch_size)
        return {tensor: value for tensor, value in zip(self.ts.input, values)}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    cfg = Config()
    print(cfg)
    cfg.from_cmd()
    print('-'*20)
    print(cfg)

    # dss = read_data_sets(cfg.sample_path)
    # app = cfg.get_app()
    # with app:
    #     app.train(dss.train, dss.validation)
    #     app.predict(dss.test)



if __name__ == '__main__':
    main()