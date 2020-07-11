# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p07_param_list_dict.py
@Description    :  
@CreateTime     :  2020/6/9 14:10
------------------------------------
@ModifyTime     :  
"""


def m1(*p):
    print(p)
    print(p[0], p[-1])
    print(len(p))


def m2(**p):
    print(p)
    for key in p:
        print(key, '=', p[key])


def m3(*args, **kargs):
    print(args)
    print(kargs)


def m4(a, b, *args, **kargs):
    print(a, b)
    print(args)
    print(kargs)


def m5(a, b, c, d):
    print(a, b, c, d)


def main():
    m1(1, 2, 3, 4)
    m2(a=3, b=2, c=9)
    m3(11, 12, 13, aa=889)
    m4(1, 2, 3, 4, 5, 6, aa=00)

    # 拆包
    lst = [1, 4, 7, 11]
    m5(*lst)
    dic = {'a': 3, 'b': 8, 'c': 0, 'd': 5}
    m5(**dic)


if __name__ == '__main__':
    main()