# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p53_sin.py
@Description    :  
@CreateTime     :  2020/7/6 10:14
------------------------------------
@ModifyTime     :  
"""
import math


def sin(a, lr=0.05, epoches=20000):
    y = lambda x: math.sin(x)
    dy_dx = lambda x: math.cos(x)
    dy = lambda x: a - y(x)
    dx = lambda x, lr: lr * dy(x) * dy_dx(x)
    x = 1.0
    for _ in range(epoches):
        x += dx(x, lr)
    return x


def main():
    print(sin(1))


if __name__ == '__main__':
    main()