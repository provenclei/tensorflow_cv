# -*- coding: utf-8 -*-
import tensorflow as tf
import argparse
import os
"""
    This is a framework which is used to implement the multiple GPU
    在43框架的基础之上
    通过端口控制训练和预测
"""


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
                parser.add_argument('--' + attr, default=value, action='store_%s' % ('false' if value else 'true'),
                                    help='Default to %s' % value)
            else:
                parser.add_argument('--' + attr, type=t, default=value, help='Default to %s' % value)
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

    def get_tensors(self):
        return Tensors(self)

    def get_sub_tensors(self, gpu_index):
        """
        Get the sub tensors for the specified gpu.
        :param gpu_index:  the index(based on zero) of the GPU
        :return: te sub tensors which has the property 'inputs'
        """
        raise Exception('The get_sub_tensors() is not defined.')

    def get_ds_train(self):
        raise Exception('get_ds_train() is not define')

    def get_ds_test(self):
        raise Exception('get_ds_test() is not define')

    def get_optimizer(self, lr):
        return tf.train.AdamOptimizer(lr)

    def get_app(self):
        return App(self)

    def train(self):
        app = self.get_app()
        with app:
            app.train(self.get_ds_train(), None)

    def test(self):
        app = self.get_app()
        with app:
            app.test(self.get_ds_train())


class Tensors:
    """
    多 GPU 计算时候，同一张量在多个GPU上，进行不同批次数据的计算（梯度等），然后进行汇总！！！
    提供 train_ops, summary, lr,
    sub_ts : 每个GPU上的张量
    sub_ts[i]: 第i个GPU上的张量{inputs(sub_ts), losses, private_tensors}
    """
    def __init__(self, config: Config):
        self.config = config
        self.sub_ts = []
        with tf.variable_scope(config.get_name()) as scope:
            # with tf.variable_scope(config.get_name(), reuse=tf.AUTO_REUSE):
            # tf.AUTO_REUSE:没有则新建，有则重用(  None:根据父范围继承  True:重用（必须都重用，没有则报错）
            for i in range(config.gpus):
                with tf.device('/gpu:%d' % i):
                    self.sub_ts.append(config.get_sub_tensors(i))
                    # 方法一  采用此方法
                    scope.reuse_variables()
                    # 方法二
                    # tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('%s_train' % config.get_name()):
            with tf.device('/gpu:0'):  # 就在GPU0上运算
                # losses属性在 MySubtensors 中定义
                # sub_ts[i] 中包含的属性：losses(可能包含多个损失), inputs 输入张量
                losses = [ts.losses for ts in self.sub_ts]  # [gpus, losses]
                self.losses = tf.reduce_mean(losses, axis=0)  # [losses]

                self.lr = tf.placeholder(tf.float32, [], name='lr')
                # opt的两个方法：
                # compute_gradients(self): Compute and store gradients of loss function for all input vectors.
                # apply_gradients(self, grads_and_vars, name=None): Apply gradients to variables.
                # 操作步骤：
                # 1. 获取多个 GPU 上的梯度
                # 2. 计算平均梯度
                # 3. 将梯度应用到变量
                opt = config.get_optimizer(self.lr)

                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    grads = self.compute_grads(opt)  # (tuple)
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
        # 只有当 运行时才发生控制依赖
        self.train_ops = [opt.apply_gradients(gs) for gs in grads]

    def get_grads_mean(self, grads, loss_idx):
        '''
        计算平均梯度
        :param grads: [gups, losses]
        :param loss_idx:
        :return:
        '''
        # 获取当前 loss 下的所有梯度
        grads = [gs[loss_idx] for gs in grads]  # 第i个梯度 [gpus, vars, 2]    2:[grads, vars]
        gpus = len(grads)

        # 获取变量列表
        vars = [pair[1] for pair in grads[0]]
        result = []
        for i, var in enumerate(vars):
            # 获取梯度
            g = grads[0][i][0]  # 0号 gpu 当前的变量
            if isinstance(g, tf.IndexedSlices):
                # 第一个 gpu 上是 IndexedSlices，其他 gpu 上均为 IndexedSlices

                # 以下方法由于生成的是张量，无法更新
                # slices = [(v / gpus, i) for v, i in zip(gs[i][0].values, gs[i][0].indices) for gs in grads]
                # values = [v for v, _ in slices]
                # indices = [i for _, i in slices]

                values = [gs[i][0].values / gpus for gs in grads]  # [gpus, -1, 200]
                values = tf.concat(values, axis=0)  # [-1, 200]
                indices = [gs[i][0].indices for gs in grads]  # [gpus, -1]
                indices = tf.concat(indices, axis=0)  # [-1]
                result.append((tf.IndexedSlices(values, indices), var))
            else:
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
        make_dirs(config.save_path)
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
        pass