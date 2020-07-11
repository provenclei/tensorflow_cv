# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p54_binary_tree.py
@Description    :  
@CreateTime     :  2020/7/6 10:53
------------------------------------
@ModifyTime     :  
"""


def binary_tree(n, buffer={}):
    if n < 2:
        return 1

    if n in buffer:
        return buffer[n]

    n -= 1
    result = 0
    for left in range(n+1):
        result += binary_tree(left, buffer) * binary_tree(n - left, buffer)
    buffer[n + 1] = result
    return result


def main():
    for i in range(1, 20 + 1):
        print("%d: \t %d" % (i, binary_tree(i)))


if __name__ == '__main__':
    main()