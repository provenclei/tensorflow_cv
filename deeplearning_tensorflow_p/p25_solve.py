# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p25_sove.py
@Description    :  
@CreateTime     :  2020/6/9 19:02
------------------------------------
@ModifyTime     :  
"""
import math


def solve_min(dy_dx, lr=0.01, epoches=2000):
    dx = lambda x, lr: -lr * dy_dx(x)
    x = 1
    for _ in range(epoches):
        x += dx(x, lr)
    return x


def solve(y, dy_dx, value, lr=0.01, epoches=2000):
    # loss = lambda x: (value - y(x)) ** 2
    dloss_dx = lambda x: 2 * (value - y(x)) * dy_dx(x)
    dx = lambda x, lr: -lr * dloss_dx(x)
    x = 1
    for _ in range(epoches):
        x += dx(x, lr)
    return x


def main():
    n = 0.5
    y = lambda x: math.sin(x)
    dy_dx = lambda x: math.cos(x)
    # loss = lambda x: (y - n) ** 2
    dloss_dx = lambda x: 2 * (y(x) - n) * dy_dx(x)
    print('arcsin(%s) = %s' % (n, solve_min(dloss_dx)*180 / math.pi))

    print('arcsin(%s) = %s' % (n, solve(y, dy_dx, 0.5)*180 / math.pi))

    n = 2
    y = lambda x: x**3
    dy_dx = lambda x: 3 * x**2
    # loss = lambda x: (n - y(x))**2
    dloss_dx = lambda x: 2 * (n - y(x)) * dy_dx(x)
    print('arcsin(%s) = %s' % (n, solve_min(dloss_dx)))

    print('arcsin(%s) = %s' % (n, solve(y, dy_dx, 2)))

    y = lambda x: -math.e ** math.sin(x)
    dy_dx = lambda x: y(x) * math.cos(x)
    result = solve_min(dy_dx)
    print('argmax=', result)
    print(y(result))


if __name__ == '__main__':
    main()