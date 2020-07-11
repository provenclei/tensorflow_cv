# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p43_framework_muti-gpu.py
@Description    :  
@CreateTime     :  2020/6/29 10:15
------------------------------------
@ModifyTime     :  多 GPU 的Framework
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


def make_dirs(path:str):
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
        return Tensors(self)

    def get_app(self):
        return App(self)

    def get_sub_tensors(self, gpu_idx):
        '''
        为每个GPU配置的张量
        :param id: gpu index
        :return: the sub tensors which has the property 'inputs'
        '''
        raise Exception('the get_sub_tensors() is not define')

    def get_optimizer(self, lr):
        return tf.train.AdamOptimizer(lr)


class Tensors:
    '''
    train_op, summary, subinputs[i]:{inputs, private tensors}
    在framework 中没有 private tensor
    '''
    def __init__(self, config: Config):
        self.config = config
        # 输入张量集合
        self.sub_ts = []  # [gpus]
        # 可训练张量，均需要设置 scope
        # with tf.variable_scope(config.get_name(), reuse=True)
        # reuse 取值为 True, None, tf.AUTO_REUSE
        # reuse = True 变量必须存在，不存在会报错
        # reuse = AUTO_REUSE  自动重用变量，但是有时会不小心重复变量名，会导致错误，所以不适用这种方法
        # reuse = None  默认为该值，从父类中寻找 reuse 的值，对于首个 CPU，如果没出现过会创建，如果没出现过，会报错！
        # 为什么不能为 False?
        # 对已有变量，重新创建同名变量，是不允许的！
        with tf.variable_scope(config.get_name()) as scope:
            for i in range(config.gpus):
                with tf.device('/gpu:%d' % i):
                    self.sub_ts.append(config.get_sub_tensors(i))
                    # tf.get_variable_scope().reuse_variables()
                    scope.reuse_variables()   # 等价

        with tf.variable_scope('%s_train' % config.get_name()):
            with tf.device('/gpu:0'):
                # 如果模型有多个 loss
                losses = [ts.losses for ts in self.sub_ts]  # [gpus, losses]
                # 平均后的 loss
                self.losses = tf.reduce_mean(losses, axis=0)  # losses
                self.lr = tf.placeholder(tf.float32, name='lr')

                # train_op 需要对梯度求平均
                # 计算梯度
                opt = config.get_optimizer(self.lr)
                # 控制依赖
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    # tf.train.AdamOptimizer().minimize() 中minimize()做的事情
                    # 返回[(梯度， 变量), (梯度，变量)]
                    grads = self.compute_grads(opt)
                    self.apply_grads(grads, opt)

            for i in range(len(losses[0])):
                # tf.summary.scalar('loss_%d' % i, self.losses[i])
                tf.summary.scalar('loss_%d' % i, self.get_loss_for_summary(self.losses[i]))
            self.summary = tf.summary.merge_all()

    def get_loss_for_summary(self, loss):
        return loss

    def compute_grads(self, opt):
        grads = [[opt.compute_gradients(loss) for loss in ts.losses] for ts in self.sub_ts]  # [gpus, losses]
        # for i in range(len(grads[0])):
        #     self.get_grads_mean(grads, i)
        return [self.get_grads_mean(grads, i) for i in range(len(grads[0]))]   # [gpus, variables]

    def apply_grads(self, grads, opt):
        self.train_ops = [opt.apply_gradients(gs) for gs in grads]

    def get_grads_mean(self, grads, loss_idx):
        # grads: [gpus, losses]
        grads = [gs[loss_idx] for gs in grads]  # [gpus]
        # 获取变量，每个 gpu 中变量相同，从第一个 gpu 中提取即可
        vars = [pair[1] for pair in grads[0]]  # (梯度，变量)，变量个数
        result = []
        for i, var in enumerate(vars):
            # 对不同变量的提取求均值，格式：[(梯度，变量), (梯度，变量)]
            result.append((tf.reduce_mean([gs[i][0] for gs in grads], axis=0), var))
        return result


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
        self.before_train()
        cfg = self.config
        ts = self.ts
        writer = tf.summary.FileWriter(cfg.logdir, self.session.graph)
        batches = ds_train.num_examples // (cfg.batch_size * cfg.gpus)

        for epoch in range(cfg.epoches):
            self.before_epoch(epoch)
            for batch in range(batches):
                self.before_batch(epoch, batch)
                # _, summary = self.session.run([ts.train_op, ts.summary], self.get_feed_dict(ds_train))
                feed_dict = self.get_feed_dict(ds_train)
                if len(ts.train_ops) == 1:
                    _, summary = self.session.run([ts.train_ops[0], ts.summary], feed_dict)
                else:
                    for train_op in ts.train_ops:
                        self.session.run(train_op, feed_dict)
                    summary = self.session.run(ts.summary, feed_dict)
                writer.add_summary(summary, epoch * batches + batch)
                self.after_batch(epoch, batch)
            print('Epoch:', epoch, flush=True)
            self.after_epoch(epoch)
        self.after_train()

    def before_train(self):
        print('Training is started!', flush=True)

    def before_epoch(self, epoch):
        pass

    def before_batch(self, epoch, batch):
        pass

    def after_train(self):
        print('Training is finished!', flush=True)

    def after_epoch(self, epoch):
        self.save()

    def after_batch(self, epoch, batch):
        pass

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('Save model into', self.config.save_path, flush=True)

    def predict(self, ds):
        pass

    def get_feed_dict(self, ds):
        result = {self.ts.lr: self.config.lr}
        # 每个 GPU 下的每个输入张量都需要有一个值
        for i in range(self.config.gpus):
            # 张量都在 inputs 里面，values 中有 xs, ys
            # next_batch 是 ds_train 的对象方法
            values = ds.next_batch(self.config.batch_size)
            for tensor, value in zip(self.ts.sub_ts[i].inputs, values):
                result[tensor] = value
        return result

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

    dss = read_data_sets(cfg.sample_path)
    app = cfg.get_app()
    with app:
        app.train(dss.train, dss.validation)
        app.predict(dss.test)


if __name__ == '__main__':
    main()