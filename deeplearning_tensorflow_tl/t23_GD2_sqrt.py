# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t23_GD2_sqrt.py
@Description    :  
@CreateTime     :  2020/6/15 18:06
------------------------------------
@ModifyTime     :  
"""


def sqrt(n, lr=0.001, epoches=2000):
    y = lambda x: x**2
    # loss = lambda x: (y(x) - n)**2
    dloss_x = lambda x: 2*(y(x) - n) * 2 * x
    dx = lambda x, lr: -lr * dloss_x(x)

    x = 1
    for _ in range(epoches):
        x += dx(x, lr)
    return x


def main():
    for i in range(11):
        print('sqrt(%s) = %f' % (i, sqrt(i)))


if __name__ == '__main__':
    main()