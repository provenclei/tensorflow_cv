# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  temp0.py
@Description    :  
@CreateTime     :  2020/7/27 11:36
------------------------------------
@ModifyTime     :  协程
"""


def m1():
    result = []
    for i in range(10):
        result.append(i)
    return result


def m():
    for i in range(10):
        yield i

def m3():
    yield 1
    yield 2
    yield 3


def main():
    for i in m():
        print(i)
    obj = m()
    print(type(obj))
    while True:
        try:
            value = next(obj)
            print(value)
        except StopIteration:
            break

    print(m3())


if __name__ == '__main__':
    main()