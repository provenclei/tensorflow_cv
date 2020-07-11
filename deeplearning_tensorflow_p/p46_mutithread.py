# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p46_mutithreading.py
@Description    :  
@CreateTime     :  2020/7/1 10:20
------------------------------------
@ModifyTime     :  
"""
import threading
import time

n = 0
lock = threading.Lock()


def add():
    global n, lock
    for _ in range(500000):
        with lock:
            n += 1

        # 等价于：
        # lock.acquire()
        # try:
        #     n += 1
        # finally:
        #     lock.release()


def sub():
    global n, lock
    for _ in range(500000):
        lock.acquire()
        n -= 1
        lock.release()


def main():
    def m():
        print('in m()')
        time.sleep(10)
    thread_add = threading.Thread(target=add)
    thread_sub = threading.Thread(target=sub)
    thread_add.start()
    thread_sub.start()

    thread_add.join()
    thread_sub.join()
    print('n = ', n, flush=True)


if __name__ == '__main__':
    main()