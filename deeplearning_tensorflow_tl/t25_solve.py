# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t25_solve.py
@Description    :  
@CreateTime     :  2020/6/15 18:13
------------------------------------
@ModifyTime     :  
"""
import math


def solve_min(loss):
    pass


def solve(y, dy_dx, lr):
    pass


def main():
    n = 0.5
    y = lambda x: math.sin(x)
    dy_dx = lambda x: math.cos(x)
    # loss = lambda x: (y - n) ** 2
    dloss_dx = lambda x: 2 * (y(x) - n) * dy_dx(x)
    print('arcsin(%s) = %s' % (n, solve_min(dloss_dx) * 180 / math.pi))
    print('arcsin(%s) = %s' % (n, solve(y, dy_dx, 0.5) * 180 / math.pi))




if __name__ == '__main__':
    main()