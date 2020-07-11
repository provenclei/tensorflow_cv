# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p09_hanoi.py
@Description    :  
@CreateTime     :  2020/6/9 14:30
------------------------------------
@ModifyTime     :  汉诺塔
"""


def hanoi(panes, src, buffer, dst):
    if panes == 1:
        print('Move %s == %d ==> %s' % (src, 1, dst))
    else:
        hanoi(panes-1, src, dst, buffer)
        print('Move %s == %d ==> %s' % (src, panes, dst))
        hanoi(panes-1, buffer, src, dst)


def main():
    hanoi(4, 'A', 'B', 'C')


if __name__ == '__main__':
    main()