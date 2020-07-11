# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t35_tensorboard.py
@Description    :  
@CreateTime     :  2020/6/19 15:06
------------------------------------
@ModifyTime     :

步骤：
1. with tf.device('/gpu:0'):

2. cfg = tf.ConfigProto()
        cfg.allow_soft_placement = True
3. 使用 ftp 工具，将代码传到 ai 服务器中
4. 使用 telnet 协议工具（PuTTy软件）控制服务器
ssh tl@192.168.2.3
5. 在管理员创建的虚拟环境下运行程序
6. 设置环境变量：CUDA_VISIBLE_DEVICES: 告诉GPU执行第几块GPU，而不是第几个！
    临时更改：
        mac中设置:
        export CUDA_VISIBLE_DEVICES=0,3
        展示：
        echo $CUDA_VISIBLE_DEVICES
7. 让系统在后台运行
    nohup python a.py &
    所有运行结果再 nohup.out 中
8. 使用 ps -ef|grep p37 查看进程
9. 使用 kill + 进程号 命令中断一个进程


在服务端启动tensorboard：
1. nohup tensorboard --port 9044
2. 创建 logdir 的目录 ./og
3. 查看 tendorboard 进程：
    ps -ef|grep tensorboard
    记录ip和端口号，在本机查阅 tensorboard
    查看日志：vi nohup.out
4.关闭：
    kill 12675
    kill -9 12675 强制终止进程，但会留下部分垃圾

计算图：(有向无环图)
如果只有一个计算图，创建两个app对象时，当给可训练张量给名字时，会报错


"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets


class Config:
    def __init__(self):
        self.save_path = './models/p35'
        self.data_path = './MNIST_data'
        self.log_path = './logs'
        self.batch_size = 256
        self.lr = 0.01
        self.epoches = 10
        self.new_model = True


class Samples:
    def __init__(self, config: Config):
        self.config = config
        self.data = read_data_sets(config.data_path)

    @property
    def num_of_valid(self):
        return self.data.validation.num_examples

    @property
    def num_of_test(self):
        return self.data.test.num_examples

    @property
    def num_of_train(self):
        return self.data.train.num_examples

    @property
    def get_train_data(self):
        return self.data.train.next_batch(self.config.batch_size)

    @property
    def get_test_data(self):
        return self.data.test.next_batch(self.config.batch_size)

    @property
    def get_valid_data(self):
        return self.data.validation.next_batch(self.config.batch_size)


class Tensors:
    def __init__(self):
        with tf.device('/gpu:0'):
            self.x = tf.placeholder(tf.float32, [None, 784], name='x')
            self.y = tf.placeholder(tf.int32, [None], name='y')

            x = tf.reshape(self.x, [-1, 28, 28, 1])
            x = tf.layers.conv2d(x, 16, 3, 1, 'same', activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 32, 3, 2, 'same', activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 3, 2, 'same', activation=tf.nn.relu)
            x = tf.layers.flatten(x)

            logits = tf.layers.dense(x, 10)
            y = tf.one_hot(self.y, 10)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y, logits))
            self.loss_summary = tf.summary.scalar('loss', self.loss)
            self.y_predict = tf.argmax(logits, axis=1, output_type=tf.int32)

            self.precise = tf.cast(tf.equal(self.y, self.y_predict), tf.float32)
            self.precise = tf.reduce_mean(self.precise)
            self.precise_summary = tf.summary.scalar('precise', self.precise)

            self.lr = tf.placeholder(tf.float32, (), name='lr')
            opt = tf.train.AdamOptimizer(self.lr)
            self.opt_train = opt.minimize(self.loss)


class MNISTApp:
    def __init__(self, config: Config):
        self.config = config
        # 创建图
        graph = tf.Graph()
        # 使用默认图
        with graph.as_default():
            self.ts = Tensors()

            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True
            # 将图添加在 session 中
            self.session = tf.Session(config=cfg, graph=graph)
            self.samples = Samples(config)
            self.saver = tf.train.Saver()

            # 变量也应该在对应的图中取寻找
            if config.new_model:
                print('加载新模型')
                self.session.run(tf.global_variables_initializer())
            else:
                try:
                    self.saver.restore(self.session, self.config.save_path)
                    print('模型已保存')
                except:
                    print('初始化模型')
                    self.session.run(tf.global_variables_initializer())

    def train(self):
        writer = tf.summary.FileWriter(self.config.log_path, self.session.graph)
        batches = self.samples.num_of_train // self.config.batch_size

        for epoche in range(self.config.epoches):
            for batch in range(batches):
                xs_train, ys_train = self.samples.get_train_data
                _, loss, loss_summary = self.session.run([self.ts.opt_train, self.ts.loss, self.ts.loss_summary], {
                    self.ts.x: xs_train,
                    self.ts.y: ys_train,
                    self.ts.lr: self.config.lr
                })
                print("epoche: %d, batch: %dth, loss: %f" % (epoche, batch, loss))
                writer.add_summary(loss_summary, epoche * batches + batch)
            # xs_valid, ys_valid = self.data.validation.next_batch(self.config.batch_size)
            xs_valid, ys_valid = self.samples.get_valid_data
            precise, precise_summary = self.session.run([self.ts.precise, self.ts.precise_summary], {
                self.ts.x: xs_valid,
                self.ts.y: ys_valid
            })
            print("precise: %f" % precise)
            writer.add_summary(precise_summary, epoche * batches)
            self.save()
        writer.close()

    def predict(self):
        precise_total = 0
        batches = self.samples.num_of_test // self.config.batch_size
        for _ in range(batches):
            xs_test, ys_test = self.samples.get_test_data
            precise = self.session.run(self.ts.precise, {
                self.ts.x: xs_test,
                self.ts.y: ys_test
            })
            precise_total += precise
        print('precise_total:', precise_total / batches)

    def save(self):
        self.saver.save(self.session, self.config.save_path)

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    cfg = Config()
    app = MNISTApp(cfg)
    with app:
        # app.train()
        app.predict()

    cfg = Config()
    app = MNISTApp(cfg)
    with app:
        # app.train()
        app.predict()


if __name__ == '__main__':
    main()