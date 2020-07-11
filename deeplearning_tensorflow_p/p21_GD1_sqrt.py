# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p21_GD1_sqrt.py
@Description    :  
@CreateTime     :  2020/6/9 16:11
------------------------------------
@ModifyTime     :  
"""


def sqrt(n):
    y = lambda x: x**2
    dy_dx = lambda x: 2*x
    dx = lambda x, lr: lr * (n - y(x)) * dy_dx(x)

    x = 1  # 初始值
    lr = 0.001
    for _ in range(2000):
        x += dx(1, lr)
    return x


def main():
    for i in range(1, 11):
        print('sqrt(%s)=%f' % (i, sqrt(i)))


if __name__ == '__main__':
    main()