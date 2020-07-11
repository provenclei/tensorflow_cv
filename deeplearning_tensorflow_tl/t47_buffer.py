# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t47_buffer.py
@Description    :  
@CreateTime     :  2020/7/5 11:08
------------------------------------
@ModifyTime     :  
"""
import threading
import time


class Buffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.total_size = buffer_size

        lock = threading.RLock()
        self.pos_lock = threading.Condition(lock)  # 是否有数据
        self.data_lock = threading.Condition(lock)  # 是否越界

    def get_size(self):
        return len(self.buffer)

    def get(self):
        '''
        从buffer中获取数据，如果buffer为空，则线程阻塞
        :return:
        '''
        with self.data_lock:
            if len(self.buffer) == 0:
                self.data_lock.wait()
            result = self.buffer[0]
            del self.buffer[0]
            self.pos_lock.notify_all()
        return result

    def put(self, data):
        '''
        向buffer中输入数据，如果满了，则线程阻塞
        :return:
        '''
        with self.pos_lock:
            if len(self.buffer) >= self.total_size:
                self.pos_lock.wait()
            self.buffer.append(data)
            self.data_lock.notify_all()


def main():
    buffer = Buffer(10)

    def get():
        for i in range(100):
            print(buffer.get())
            time.sleep(0.1)

    def put():
        for i in range(100):
            buffer.put(i)

    th_put = threading.Thread(target=put, daemon=True)
    th_get = threading.Thread(target=get, daemon=True)

    th_get.start()
    th_put.start()
    th_put.join()
    th_get.join()


if __name__ == '__main__':
    main()