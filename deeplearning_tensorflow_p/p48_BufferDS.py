# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p48_BufferDS.py
@Description    :  
@CreateTime     :  2020/7/2 09:50
------------------------------------
@ModifyTime     :  装修设计模式
"""
from p47_buffer import Buffer
from p45_celeba import CelebA

import threading


class BufferDS:
    def __init__(self, buffer_size, ds, batch_size):
        # ds.num_examples, ds.next_batch(batch_size)
        self.ds = ds
        self.buffer = Buffer(buffer_size)
        self.batch_size = batch_size
        self.reader = threading.Thread(target=self.read, daemon=True)  # 使用傀儡线程，主线程结束，自动销毁
        self.reader.start()

    def read(self):
        while True:
            data = self.ds.next_batch(self.batch_size)
            self.buffer.put(data)

    def next_batch(self, batch_size):
        '''
        framework 中需要 batch_size 参数，必须要传入，否则报错
        :param batch_size:
        :return:
        '''
        return self.buffer.get()

    @property
    def num_examples(self):
        return self.ds.num_examples


def main():
    pass


if __name__ == '__main__':
    main()