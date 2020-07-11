# This is a framework which is used to implement the multiple GPU
import tensorflow as tf
import argparse
import numpy as np
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import os


def get_gpus():  # gpu数量
    value = os.getenv('CUDA_VISIBLE_DEVICES', '0')  # 获取环境变量
    value = value.split(',')
    return len(value)


def make_dirs(path: str):
    pos = path.rfind(os.sep)
    if pos < 0:
        raise Exception('Can not find the directory from the path', path)
    path = path[: pos]
    os.makedirs(path, exist_ok=True)


class Config:
    def __init__(self):
        self.lr = 0.01
        self.epoches = 2000
        self.batch_size = 500
        self.new_model = False
        self.gpus = get_gpus()

        self.save_path = './models/{name}/{name}'.format(name=self.get_name())
        self.sample_path = None
        self.log_dir = './logs/{name}'.format(name=self.get_name())

    def get_name(self):
        raise Exception('get_name() is not re-defined.')

    def __repr__(self):
        attrs = self.get_attrs()
        result = ['%s = %s' % (key, attrs[key]) for key in attrs]
        return ', '.join(result)

    def get_attrs(self):
        result = {}
        for attr in dir(self):
            if attr.startswith('_'):
                continue
            value = getattr(self, attr)
            if value is None or type(value) in (int, float, bool, str):
                result[attr] = value
        return result

    def from_cmd(self):
        parser = argparse.ArgumentParser()
        attrs = self.get_attrs()
        for attr in attrs:
            value = attrs[attr]
            t = type(value)
            if t == bool:
                parser.add_argument('--' + attr, default=value, action='store_%s' % ('false' if value else 'true'), help='Default to %s' % value)
            else:
                parser.add_argument('--' + attr, type=t, default=value, help='Default to %s' % value)
        a = parser.parse_args()
        for attr in attrs:
            setattr(self, attr, getattr(a, attr))

    def get_tensors(self):
        return Tensors(self)

    def get_app(self):
        return App(self)

    def get_sub_tensors(self, gpu_index):
        """
        Get the sub tensors for the specified gpu.
        :param gpu_index:  the index(based on zero) of the GPU
        :return: te sub tensors which has the property 'inputs'
        """
        raise Exception('The get_sub_tensors() is not defined.')

    def get_optimizer(self, lr):
        return tf.train.AdamOptimizer(lr)


class Tensors:
    """
    提供 train_ops, summary, lr, sub_ts[i]: {inputs, losses, private_tensors}
    """
    def __init__(self, config: Config):
        self.config = config
        self.sub_ts = []
        with tf.variable_scope(config.get_name()) as scope:
            # with tf.variable_scope(config.get_name(), reuse=tf.AUTO_REUSE):
            # tf.AUTO_REUSE:没有则新建，有则重用(  None:根据父范围继承  True:重用（必须都重用，没有则报错））
            for i in range(config.gpus):
                with tf.device('/gpu:%d' % i):
                    self.sub_ts.append(config.get_sub_tensors(i))
                    # 方法一  采用此方法
                    scope.reuse_variables()
                    # 方法二
                    # tf.get_variable_scope().reuse_variables()
        with tf.device('/gpu:0'):  # 就在GPU0上运算
            with tf.variable_scope('%s_train' % config.get_name()):
                losses = [ts.losses for ts in self.sub_ts]   # [gpus, losses]
                self.losses = tf.reduce_mean(losses, axis=0)  # [losses]

                self.lr = tf.placeholder(tf.float32, [], name='lr')
                opt = config.get_optimizer(self.lr)
                # opt.minimize()

                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    grads = self.compute_grads(opt)  # (tuple)
                    self.apply_grads(grads, opt)

            for i in range(len(losses[0])):
                tf.summary.scalar('loss_%d' % i, self.get_loss_for_summary(self.losses[i]))
            self.summary = tf.summary.merge_all()

    def get_loss_for_summary(self, loss):
        return loss

    def compute_grads(self, opt):
        # [opt.compute_gradients(loss) for loss in ts.losses]
        grads = [[opt.compute_gradients(loss) for loss in ts.losses] for ts in self.sub_ts]  # [gpus, losses]
        return [self.get_grads_mean(grads, i) for i in range(len(grads[0]))]

    def apply_grads(self, grads, opt):
        self.train_ops = [opt.apply_gradients(gs) for gs in grads]

    def get_grads_mean(self, grads, loss_idx):
        #  grads: [gups, losses]
        grads = [gs[loss_idx] for gs in grads]  # 第i个梯度 [gpus]
        vars = [pair[1] for pair in grads[0]]  # 变量
        result = []
        for i, var in enumerate(vars):
            result.append((tf.reduce_mean([gs[i][0] for gs in grads], axis=0), var))
        return result


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
                print('use a new model')
                self.session.run(tf.global_variables_initializer())
            else:
                try:
                    self.saver.restore(self.session, config.save_path)
                    print('Restore the model from %s successfully' % config.save_path)
                except:
                    print('Fail to restore the model from %s, use a new model instead' % config.save_path)
                    self.session.run(tf.global_variables_initializer())

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def train(self, ds_train, ds_validation):
        self.before_train()
        config = self.config
        ts = self.ts
        writer = tf.summary.FileWriter(config.log_dir, self.session.graph)  # 唯一在训练中定义的对象
        batches = ds_train.num_examples // (config.batch_size * config.gpus)

        for epoch in range(config.epoches):
            self.before_epoch(epoch)
            for batch in range(batches):
                self.before_batch(epoch, batch)
                feed_dict = self.get_feed_dict(ds_train)
                if len(ts.train_ops) == 1:
                    _, summary = self.session.run([ts.train_ops[0], ts.summary], feed_dict)
                else:
                    for train_op in ts.train_ops:
                        self.session.run(train_op, feed_dict)
                    summary = self.session.run(ts.summary, feed_dict)
                writer.add_summary(summary, epoch * batches + batch)
                self.after_batch(epoch, batch)
            print('Epoch: ', epoch, flush=True)
            self.after_epoch(epoch)
        self.after_train()

    def before_train(self):
        print('Trainging is started!', flush=True)

    def before_epoch(self, epoch):
        pass

    def before_batch(self, epoch, batch):
        pass

    def after_train(self):
        print('Training is finished!', flush=True)

    def after_batch(self, epoch, batch):
        pass

    def after_epoch(self, epoch):
        self.save()

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('Save the model into %s ' % self.config.save_path, flush=True)

    def get_feed_dict(self, ds):
        result = {self.ts.lr: self.config.lr}
        for i in range(self.config.gpus):
            values = ds.next_batch(self.config.batch_size)
            for tensor, value in zip(self.ts.sub_ts[i].inputs, values):
                result[tensor] = value
        return result

    def test(self, ds_test):
        pass


if __name__ == '__main__':
    class MyConfig(Config):
        def get_name(self):
            return 'test'
    cfg = MyConfig()
    cfg.from_cmd()
    print('-' * 100)
    print(cfg)
    dss = read_data_sets('./MNIST_data')
    app = cfg.get_app()
    with app:
        app.train(dss.train, dss.validation)
        app.test(dss.test)
