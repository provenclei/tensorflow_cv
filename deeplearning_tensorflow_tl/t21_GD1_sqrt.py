# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t21_GD1_sqrt.py
@Description    :  
@CreateTime     :  2020/6/15 18:01
------------------------------------
@ModifyTime     :  
"""


def sqrt(n):
    y = lambda x: x**2
    dy_dx = lambda x: 2*x
    dx = lambda x, lr: lr * (n - y(x)) * dy_dx(x)

    x = 1
    lr = 0.01
    for _ in range(2000):
        x += dx(x, lr)
    return x


if __name__ == '__main__':
    for i in range(1, 11):
        print('sqrt(%s)=%f' % (i, sqrt(i)))