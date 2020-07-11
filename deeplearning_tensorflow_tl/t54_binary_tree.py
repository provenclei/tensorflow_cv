# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t54_binary_tree.py
@Description    :  
@CreateTime     :  2020/7/6 13:44
------------------------------------
@ModifyTime     :  已知节点个数，生成的二叉树有几种？
"""


def binary_tree(n, buffer={}):
    if n < 2:
        return 1
    if n in buffer:
        return buffer[n]
    result = 0
    n -= 1  # 根节点
    for i in range(n + 1):
        result += binary_tree(i, buffer) * binary_tree(n-i, buffer)
    buffer[n + 1] = result
    return result


def main():
    for i in range(1, 15 + 1):
        print("%d: \t %d" % (i, binary_tree(i)))


if __name__ == '__main__':
    main()