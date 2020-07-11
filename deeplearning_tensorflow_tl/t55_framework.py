# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t55_framework.py
@Description    :  
@CreateTime     :  2020/7/9 15:44
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
import os
import argparse


def get_gpus():
    result = os.getenv('NVIDA INVISIBLE DEVICES', '0')
    return len(result.split(','))


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
        raise Exception('未重写 get_name() 方法')

    def from_cmd(self):
        parser = argparse.ArgumentParser()
        attrs = self.get_attrs()
        for attr in attrs:
            value = attrs[attr]
            t = type(attr)
            if t == bool:
                paser.add_argument('--'+attr, default=value, help='default is %s' % value,
                                   action='store_%s' % ('false' if value else 'true'))
            else:
                parser.add_argument('--'+attr, type=t, default=value, help='default is %s' % attr)
        parser.add_argument('--call', type=str, default='train', help='call method, by default call train()')
        a = parser.parse_args()
        for attr in attrs:
            setattr(self, attr, getattr(a, attr))
        self.call(a.call)

    def call(self, name):
        if name == 'train':
            self.train()
        elif name == 'test':
            self.test()
        else:
            print('unknow method name' + name, flush=True)

    def get_attrs(self):
        result = {}
        for attr in dir(self):
            if attr.startswith('_'):
                continue
            value = getattr(self, attr)
            if value is None or type(value) in (int, float, bool, str):
                result[attr] = value
        return result

    def __repr__(self):
        attrs = self.get_attrs()
        result = ['%s = %s' % (key, attrs[key]) for key in attrs]
        return ', '.join(result)

    def get_tensors(self):
        return Tensors(self)

    def get_sub_tensors(self, gpu_idx):
        raise Exception('未重定义 get_sub_tensors() 方法')

    def get_app(self):
        return App(self)

    def get_ds_train(self):
        raise Exception('get_ds_train() is not define')

    def get_ds_test(self):
        raise Exception('get_ds_test() is not define')

    def train(self):
        app = self.get_app()
        with app:
            app.train(self.get_ds_train())

    def test(self):
        with self.get_app() as app:
            app.test(self.get_ds_test())

    def get_optimizer(self, lr):
        return tf.train.AdamOptimizer(lr)


class Tensors:
    def __init__(self, config: Config):
        self.config = config
        self.sub_ts = []
        with tf.variable_scope(config.get_name()) as scope:
            for i in range(config.gpus):
                self.sub_ts.append(config.get_sub_tensors(i))
                scope.reuse_variables()
        with tf.variable_scope('%s_train' %config.get_name()):
            with tf.device('/gpu:0'):
                losses = [ts.losses for ts in self.sub_ts]
                self.losses = tf.reduce_mean(losses, axis=0)
                self.lr = tf.placeholder(tf.float32, [], name='lr')
                opt = config.get_optimizer(self.lr)
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    grads = self.compute_grads(opt)
                    self.apply_grads(grads, opt)
            for i in range(len(losses[0])):
                tf.summary.scalar('loss_%d' % i, self.get_loss_for_summary(self.losses[i]))
            self.summary = tf.summary.merge_all()

    def get_loss_for_summary(self, loss):
        return loss

    def compute_grads(self, opt):
        grads = [[opt.compute_gradients(loss) for loss in ts.losses] for ts in self.sub_ts]  # [gpus, losses]
        return [self.get_grads_mean(grads, i) for i in range(len(grads[0]))]

    def apply_grads(self, grads, opt):
        self.train_ops = [opt.apply_gradients(gs) for gs in grads]

    def get_grads_mean(self, grads, loss_idx):
        grads = [gs[loss_idx] for gs in grads]
        vars = [pair[1] for pair in grads[0]]
        result = []
        for i, var in enumerate(vars):
            result.append((tf.reduce_mean([gs[i][0] for gs in grads], axis=0), var))
        return result


class App:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True
            self.session = tf.Session(graph=graph, config=cfg)
            self.ts = Tensors(self.config)
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

    def train(self, ds_train):
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
        print('Epoch: ', epoch, flush=True)
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
        print('test')
        pass