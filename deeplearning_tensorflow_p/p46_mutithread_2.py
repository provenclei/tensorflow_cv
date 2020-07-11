# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p46_mutithread_2.py
@Description    :  
@CreateTime     :  2020/7/2 16:00
------------------------------------
@ModifyTime     :  多线程的封装和属性
"""
import threading
from time import sleep, ctime


class MyThread(threading.Thread):
    def run(self) -> None:
        for i in range(3):
            sleep(1)
            msg = "I'm " + self.name + ' @ ' + str(i)  # name属性中保存的是当前线程的名字
            print(msg)


def sing():
    for i in range(3):
        print("正在唱歌...%d"%i)
        sleep(1)


def dance():
    for i in range(3):
        print("正在跳舞...%d" % i)
        sleep(1)


def test():
    for i in range(5):
        t = MyThread()
        t.start()


if __name__ == '__main__':
    print('---开始---:%s'%ctime())

    t1 = threading.Thread(target=sing)
    t2 = threading.Thread(target=dance)

    t1.start()
    t2.start()

    # sleep(5) # 屏蔽此行代码，试试看，程序是否会立马结束？
    # print('---结束---:%s'%ctime())

    while True:
        length = len(threading.enumerate())
        print('当前线程个数：', length)
        # 主线程
        if length <= 1:
            break
        sleep(0.3)

    test()