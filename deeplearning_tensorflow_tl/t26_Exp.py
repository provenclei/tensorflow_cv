# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t26_Exp.py
@Description    :  
@CreateTime     :  2020/6/15 17:29
------------------------------------
@ModifyTime     :  
"""
import math


class Exp:
    def eval(self):
        pass

    def deriv(self):
        pass

    def simplify(self):
        return self

    def __neg__(self):
        return Neg(self).simplify()


class Neg(Exp):
    def __init__(self, value):
        self.value = value

    def eval(self, **values):
        return -self.value.eval(**values)

    def deriv(self, x):
        return -self.deriv(x)

    def simplify(self):
        if isinstance(self.value, Const):
            return Const(-self.value.eval())
        return self

    def __repr__(self):
        return '(-%s)' % self.value


class Const(Exp):
    def __init__(self, value):
        self.value = value

    def eval(self, **values):
        return self.value

    def deriv(self, x):
        return Const(0)

    def __repr__(self):
        return str(self.value)


class Variable(Exp):
    def __init__(self, name):
        # 变量名
        self.name = name

    def eval(self, **values):
        if self.name in values:
            return values[self.name]
        else:
            raise Exception('Variable %s is not find' % self.name)

    def deriv(self, x):
        name = _get_name(x)
        return Const(1 if name == self.name else 0)

    def __repr__(self):
        return self.name


def _get_name(x):
    if isinstance(x, Variable):
        return x.name
    if isinstance(x, str):
        return x
    raise Exception('%x can not be used to get derivant from an expression' % x)


def main():
    pass


if __name__ == '__main__':
    main()