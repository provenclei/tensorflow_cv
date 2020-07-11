# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p47_buffer.py
@Description    :  
@CreateTime     :  2020/7/1 11:23
------------------------------------
@ModifyTime     :
1. 生产者消费者问题
2. 读写问题 （文章可以被多人读取，正在写文章不能受读者干扰）
3. 舞会问题 （男女成对进入会场）
4. 水分子问题 （H*2 + O 也是一种舞会问题）
5. 红绿灯问题 （单向车道，左右两边各有一个红绿灯，保持车辆畅通）
6. 电梯问题
"""
import threading


class Buffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        # self.pos = 0  # 指向当前数据的指针
        lock = threading.RLock()  # 工厂方法，可重入锁
        self.has_data = threading.Condition(lock)
        self.has_pos = threading.Condition(lock)   # 必须依赖同一把锁

    def get_size(self):
        return self.size

    def get(self):
        '''
        get data from this buffer.
        if buffer is empty then the current thread is blocked until there is
        at least one data available
        :return:
        '''
        with self.has_data:
            while len(self.buffer) == 0:
                self.has_data.wait()
            result = self.buffer[0]
            del self.buffer[0]
            self.has_pos.notify_all()
        # result = self.buffer[self.pos]
        # self.pos = (self.pos + 1) % len(self.buffer)
        return result

    def put(self, data):
        '''
        put the data into this buffer.
        the current thread is blocked if the buffer is full
        :param data:
        :return:
        '''
        with self.has_pos:
            while len(self.buffer) >= self.size:
                self.has_pos.wait()
            self.buffer.append(data)
            self.has_data.notify_all()


def main():
    import time
    buffer = Buffer(10)

    def get():
        for _ in range(10000):
            print(buffer.get())
            time.sleep(0.01)

    def put():
        for i in range(10000):
            buffer.put(i)

    # daemon；构造一个傀儡线程，主线程停止时，傀儡线程不能独立存在。
    th_put = threading.Thread(target=put, daemon=True)
    th_get = threading.Thread(target=get, daemon=True)

    th_put.start()
    th_get.start()
    th_put.join()
    th_get.join()


if __name__ == '__main__':
    main()