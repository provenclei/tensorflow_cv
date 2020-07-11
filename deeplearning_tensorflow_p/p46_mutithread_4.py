# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p46_mutithread_4.py
@Description    :  
@CreateTime     :  2020/7/3 14:32
------------------------------------
@ModifyTime     :
                使用互斥锁实现多线程对话
"""
import threading


class XiaoAi(threading.Thread):
    def __init__(self, lock, dict):
        super().__init__(name="小爱同学")
        self.lock = lock
        self.act_dict = dict  # 1

    def wait(self):
        '''
        小爱同学需要被激活，其他线程来争夺资源
        :return:
        '''
        while True:
            if self.act_dict['activate']:
                break

    def notify(self):
        '''
        小爱同学的线程需要被锁，通知线程去争夺其他资源
        :return:
        '''
        with self.lock:
            self.act_dict['activate'] = 0

    def run(self):
        self.wait()
        print("{} : 在".format(self.name))
        self.notify()

        self.wait()
        print("{} : 好啊".format(self.name))
        self.notify()


class TianMao(threading.Thread):
    def __init__(self, lock, dict):
        super().__init__(name="天猫精灵")
        self.lock = lock
        self.act_dict = dict  # 0

    def wait(self):
        while True:
            if not self.act_dict['activate']:
                break

    def notify(self):
        with self.lock:
            self.act_dict['activate'] = 1

    def run(self):
        self.wait()
        print("{} : 小爱同学".format(self.name))
        self.notify()

        self.wait()
        print("{} : 我们来对古诗吧".format(self.name))
        self.notify()


def main():
    lock = threading.Lock()
    # 0 : 天猫
    # 1 : 小爱
    dict = {'activate': 0}  # 需要使用可变类型！！！
    xa = XiaoAi(lock, dict)
    tm = TianMao(lock, dict)

    tm.start()
    xa.start()


if __name__ == "__main__":
    main()

