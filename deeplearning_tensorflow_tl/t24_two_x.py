# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t24_two_x.py
@Description    :  
@CreateTime     :  2020/6/15 18:09
------------------------------------
@ModifyTime     :  多元函数最值
"""


def solve(lr=0.01, epoches=2000):
    y = lambda x1, x2: (x1 - 3) ** 2 + (x2 + 4) ** 4
    dy_x1 = lambda x1, x2: 2 * (x1 - 3)
    dy_x2 = lambda x1, x2: 2 * (x2 + 4)
    dx1 = lambda x1, x2, lr: -lr * dy_x1(x1, x2)
    dx2 = lambda x1, x2, lr: -lr * dy_x2(x1, x2)

    x1, x2 = 1, 1
    for _ in range(epoches):
        x1 += dx1(x1, x2, lr)
        x2 += dx2(x1, x2, lr)
    return x1, x2


def main():
    print('x1, x2=', solve())


if __name__ == '__main__':
    main()