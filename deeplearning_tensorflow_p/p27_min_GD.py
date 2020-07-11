# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p27_min_GD.py
@Description    :  
@CreateTime     :  2020/6/11 13:16
------------------------------------
@ModifyTime     :  搭建一个求导数的通用平台
"""
import p26_Exp as ep


def argmin(exp, x, lr=0.01, epoches=2000):
    d = exp.deriv(x)
    x = 1.0

    for _ in range(epoches):
        dx = -lr * d.eval(x=x)
        x += dx
    return x


if __name__ == '__main__':
    x = ep.Variable('x')
    y = ep.Variable('y')
    while True:
        exp = input('>exp=?')
        if len(exp) == 0:
            break
        print(exp)
        exp = eval(exp)
        print('deriv = ', exp.deriv(x))
        x_v = argmin(exp, x)
        print('x=', x_v)
        print('min=', exp.eval(x=x_v))