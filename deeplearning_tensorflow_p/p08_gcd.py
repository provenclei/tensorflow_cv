# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p08_gdc.py
@Description    :  
@CreateTime     :  2020/6/9 13:35
------------------------------------
@ModifyTime     : 最大公约数：辗转相除
"""


def gcd(a, b):
    a = -a if a < 0 else a
    b = -b if b < 0 else b
    if a < b:
        a, b = b, a
    while b > 0:
        a, b = b, a % b
    return a


def main():
    for i in range(1, 21):
        for j in range(1, 21):
            print('gcd（%d, %d）=%d' % (i, j, gcd(i, j)))

    cd(4, 16)


if __name__ == '__main__':
    main()