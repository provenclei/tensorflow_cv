# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p46_mutithread_5.py
@Description    :  
@CreateTime     :  2020/7/3 17:00
------------------------------------
@ModifyTime     :  使用条件变量 Condition 实现多线程对话

wait([timeout]): 调用这个方法将使线程进入 Condition 的等待池等待通知，并释放锁。
                使用前 线程必须已获得锁定，否则将抛出异常。
notify(): 调用这个方法将从等待池挑选一个线程并通知，收到通知的线程将自动调用acquire()尝试获得锁定（进入锁定池）；
            其他线程仍然在等待池中。调用这个方法不会释放锁定。
            使用前 线程必须已获得锁定，否则将抛出异常。
notifyAll(): 调用这个方法将通知等待池中所有的线程，这些线程都将进入锁定池尝试获得锁定。
            调用这个方法不会释放锁定。使用前线程必须已获得锁定，否则将抛出异常。
"""
import threading


class XiaoAi(threading.Thread):
    def __init__(self, condition):
        super().__init__(name="小爱同学")
        self.condition = condition

    def run(self):
        # self.condition.acquire()

        with self.condition:
            self.condition.wait()
            print("{} : 在".format(self.name))
            self.condition.notify()

            self.condition.wait()
            print("{} : 好啊".format(self.name))
            self.condition.notify()

            self.condition.wait()
            print("{} : 司马光砸缸".format(self.name))
            self.condition.notify()

        # self.condition.release()


class TianMao(threading.Thread):
    def __init__(self, condition):
        super().__init__(name="天猫精灵")
        self.condition = condition

    def run(self):
        # self.condition.acquire()

        with self.condition:
            print("{} : 小爱同学".format(self.name))
            self.condition.notify()

            self.condition.wait()
            print("{} : 我们来对古诗吧".format(self.name))
            self.condition.notify()

            self.condition.wait()
            print("{} : 窗前明月光".format(self.name))
            self.condition.notify()
            # self.condition.wait()

        # self.condition.release()


def main():
    condition = threading.Condition()
    xa = XiaoAi(condition)
    tm = TianMao(condition)

    # 死锁
    # tm.start()
    # xa.start()

    xa.start()
    tm.start()


if __name__ == "__main__":
    main()