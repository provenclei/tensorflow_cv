# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p10_rabbit.py
@Description    :  
@CreateTime     :  2020/6/9 14:57
------------------------------------
@ModifyTime     :  
"""


def get_rabbit(n):
    if n < 2:
        return 1
    return get_rabbit(n-1) + get_rabbit(n-2)


def main():
    for i in range(21):
        print('%d: %10d' % (i, get_rabbit(i)))


if __name__ == '__main__':
    main()