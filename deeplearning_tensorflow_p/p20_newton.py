# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p_20_newton.py
@Description    :  
@CreateTime     :  2020/6/9 15:33
------------------------------------
@ModifyTime     :  
"""


def sqrt(n):
    x = 1
    for _ in range(5):
        x = (n + x**2)/(2*x)
    return x


def main():
    for i in range(1, 11):
        print('sqrt(%s)=%f' % (i, sqrt(i)))


if __name__ == '__main__':
    main()