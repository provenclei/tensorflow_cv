# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t53_sin.py
@Description    :  
@CreateTime     :  2020/7/6 13:44
------------------------------------
@ModifyTime     :  sin x = 1, 求 x
            使用梯度下降求固定取值的函数的自变量取值
"""
import math


def sin(n, lr=0.05, epoches=2000):
    dy_dx = lambda x: math.cos(x)
    dy = lambda x: n - math.sin(x)
    dx = lambda x, lr: lr * dy_dx(x) * dy(x)

    x = 0.2
    for _ in range(epoches):
        x += dx(x, lr)
    return x


def main():
    print(sin(1))


if __name__ == '__main__':
    main()