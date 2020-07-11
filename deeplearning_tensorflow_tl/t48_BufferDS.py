# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t48_BufferDS.py
@Description    :  
@CreateTime     :  2020/7/5 14:42
------------------------------------
@ModifyTime     :  
"""
from t45_celeba import CelebA
from t47_buffer import Buffer
import threading


class BufferDS:
    def __init__(self, buffer_size, ds, batch_size):
        self.ds = ds
        self.buffer = Buffer(buffer_size)
        self.batch_size = batch_size
        self.reader = threading.Thread(target=self.read)
        self.reader.start()

    def read(self):
        while True:
            data = self.ds.next_batch(self.batch_size)
            self.buffer.put(data)

    def next_batch(self, batch_size):
        return self.buffer.get()

    @property
    def num_examples(self):
        return self.ds.num_examples
