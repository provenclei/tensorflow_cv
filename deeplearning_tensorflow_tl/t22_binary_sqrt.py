# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t22_binary_sqrt.py
@Description    :  
@CreateTime     :  2020/6/15 17:57
------------------------------------
@ModifyTime     :  
"""


def sqrt(n, eps=1e-4):
    xa = 0
    xb = (n+1) ** 2

    while abs(xa - xb) > eps:
        xm = (xa + xb) / 2
        if xm * xm >= n:
            xb = xm
        else:
            xa = xm
    return xa


def main():
    for i in range(1, 11):
        print('sqrt(%s) = %f' % (i, sqrt(i)))


if __name__ == '__main__':
    main()