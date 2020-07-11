# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p26.py
@Description    :  
@CreateTime     :  2020/6/9 22:49
------------------------------------
@ModifyTime     :  
"""
import math


class Exp:
    def eval(self, **values):
        '''
        对表达式进行求值
        :param values:
        :return:
        '''
        pass

    def deriv(self, x):
        '''
        求偏导
        :param x:
        :return:
        '''
        pass

    def simplify(self):
        '''
        将当前表达式转化为更简单的表达式
        :return:
        '''
        return self

    def __add__(self, other):
        '''
        运算符重载
        to_exp：如果 other 为常数，需要转为 Const 类型
                如果为 Exp 类型，直接返回
        :param other: self + other
        :return:
        '''
        return Add(self, to_exp(other)).simplify()

    def __radd__(self, other):
        '''
        other + self
        :param other:
        :return:
        '''
        return Add(to_exp(other), self).simplify()

    def __sub__(self, other):
        return Sub(self, to_exp(other)).simplify()

    def __rsub__(self, other):
        return Sub(to_exp(other), self).simplify()

    def __mul__(self, other):
        return Mul(self, to_exp(other)).simplify()

    def __rmul__(self, other):
        return Mul(to_exp(other), self).simplify()

    def __truediv__(self, other):
        return TrueDiv(self, to_exp(other)).simplify()

    def __rtruediv__(self, other):
        return TrueDiv(to_exp(other), self).simplify()

    def __neg__(self):
        return Neg(self).simplify()

    def __pow__(self, power, modulo=None):
        return Pow(self, to_exp(power)).simplify()

    def __rpow__(self, other):
        return Pow(to_exp(other), self).simplify()


class Neg(Exp):
    def __init__(self, value):
        self.value = value

    def eval(self, **values):
        '''
        因为要带入参数计算，所以需要对参数进行拆包后再使用eval
        :param values:
        :return:
        '''
        return -self.value.eval(**values)

    def deriv(self, x):
        # return Neg(self.value.deriv(x)).simplify()
        # 等价于
        return - self.value.deriv(x)

    def simplify(self):
        if isinstance(self.value, Const):
            return Const(-self.value.eval())
        return self

    def __repr__(self):
        '''
        用来显示表达式
        :return:
        '''
        return '(-%s)' % self.value


def to_exp(value):
    if isinstance(value, Exp):
        return value
    elif type(value) in (int, float):
        return Const(value)
    else:
        raise Exception('Can not convert %s into Exp' % value)


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


class Add(Exp):
    def __init__(self, right: Exp, left: Exp):
        self.right = right
        self.left = left

    def eval(self, **values):
        return self.left.eval(**values) + self.right.eval(**values)

    def deriv(self, x):
        # 偏导的加法等于加法的偏导
        return self.left.deriv(x) + self.right.deriv(x)

    def simplify(self):
        left = self.left
        right = self.right
        if isinstance(left, Const):  # 1 + x
            if left.value == 0:
                return right
            if isinstance(right, Const):
                return Const(left.value + right.value)
        elif isinstance(right, Const):  # x + 1
            if right.value == 0:
                return left
        return self

    def __repr__(self):
        return "(%s + %s)" % (self.left, self.right)


class Sub(Exp):
    def __init__(self, left: Exp, right: Exp):
        self.left, self.right = left, right

    def eval(self, **values):
        return self.left.eval(**values) - self.right.eval(**values)

    def deriv(self, x):
        return self.left.deriv(x) - self.right.deriv(x)

    def simplify(self):
        left, right = self.left, self.right
        if isinstance(left, Const):
            if left.value == 0:
                return -right
            if isinstance(right, Const):
                return Const(left.value - right.value)
        elif isinstance(right, Const) and right.value == 0:
            return left
        return self

    def __repr__(self):
        return "(%s - %s)" % (self.left, self.right)


class Mul(Exp):
    def __init__(self, left: Exp, right: Exp):
        self.left, self.right = left, right

    def eval(self, **values):
        return self.left.eval(**values) * self.right.eval(**values)

    def deriv(self, x):
        u, v  = self.left, self.right
        #  (uv)' = u'v + uv'
        return u.deriv(x) * v + u * v.deriv(x)

    def simplify(self):
        left = self.left
        right = self.right
        if isinstance(left, Const):
            if left.value == 0:
                return Const(0)
            elif left.value == 1:
                return right
            if isinstance(right, Const):
                return Const(left.value * right.value)
        elif isinstance(right, Const):
            if right.value == 0:
                return Const(0)
            elif right.value == 1:
                return left
        return self

    def __repr__(self):
        return "(%s * %s)" % (self.left, self.right)


class TrueDiv(Exp):
    def __init__(self, left: Exp, right: Exp):
        self.left, self.right = left, right

    def eval(self, **values):
        return self.left.eval(**values) / self.right.eval(**values)

    def deriv(self, x):
        u, v = self.left, self.right
        # (u/v)' = (u'v - uv')/v**2
        return (u.deriv(x) * v - u * v.deriv(x)) / v**2

    def simplify(self):
        left, right = self.left, self.right
        if isinstance(left, Const):
            if left.value == 0:
                return Const(0)
            if isinstance(right, Const):
                return Const(left.value / right.value)
        elif isinstance(right, Const):
            if right.value == 0:
                raise Exception('Divided by zero!')
            elif right.value == 1:
                return left
        return self

    def __repr__(self):
        return "(%s / %s)" % (self.left, self.right)


e = Const(math.e)


def log(value, base=e):
    return Log(value, base)


class Log(Exp):
    def __init__(self, value, base):
        self.value = value
        self.base = base

    def simplify(self):
        if isinstance(self.value, Const):
            if self.value.value == 1:
                return Const(0)
            if isinstance(self.base, Const):
                return Const(math.log(self.value.value, self.base.value))
        return self

    def __repr__(self):
        return 'log(%s)(%s)' % (self.base, self.value)

    def deriv(self, x):
        u, v = self.value, self.base
        #  (log(u, v))' = (u' * ln(v)/u - v' * ln(u)/v) / (ln(v)**2)
        result = u.deriv(x) * log(v) / u - v.deriv(x) * log(u) / v
        return result / log(v) ** 2

    def eval(self, **values):
        return math.log(self.value.eval(**values), self.base.eval(**values))


class Pow(Exp):
    def __init__(self, base, power):
        self.base = base
        self.power = power

    def eval(self, **values):
        return self.base.eval(**values) ** self.power.eval(**values)

    def simplify(self):
        if isinstance(self.power, Const):
            if self.power.value == 0:
                return Const(1)
            if self.power.value == 1:
                return self.base
            if isinstance(self.base, Const):
                return Const(self.base.value ** self.power.value)
        elif isinstance(self.base, Const) and self.base.value in (0, 1):
            return Const(self.base.value)
        return self

    def deriv(self, x):
        # (u**v)' = y * v' * ln(u) + v * u**(v-1) * u'
        u, v = self.base, self.power
        return self * v.deriv(x) * log(u) + v * u**(v-1) * u.deriv(x)

    def __repr__(self):
        return '(%s ** %s)' % (self.base, self.power)


class Sin(Exp):
    def __init__(self, value):
        self.value = value

    def eval(self, **values):
        return math.sin(self.value.eval(**values))

    def simplify(self):
        if isinstance(self.value, Const):
            return Const(math.sin(self.value.value))
        return self

    def deriv(self, x):
        return cos(self.value) * self.value.deriv(x)

    def __repr__(self):  # representation
        return 'sin(%s)' % self.value


def sin(value):
    return Sin(to_exp(value)).simplify()


def main():
    c1 = Const(1)
    c2 = Const(2.311)
    print(c1.eval(), c2.eval())
    # 重写repr后，结果变为字符串，否则为对象
    print(c1, c2)

    x = Variable('x')
    y = Variable('y')
    print(x.eval(x=111), y.eval(x=22.44, y=33.222))

    print('c1 + c2 = ', (c1 + c2).eval())
    print('c1 + x = ', (c1 + x).eval(x=2))
    print((c1 * (x + y)).eval(x=2, y=8))

    print('c1 + c2 = ', c1 + c2)
    print('c1 + c2 * c1 / 5 = ', c1 + c2 * c1 / 5)
    print('(c1 + c2) * c2 = ', (c1 + c2) * c2)

    print(c1 + 3)
    print(3 + x)
    print(3 - c1)

    print(-c1)
    print(3-3-c1)

    print((3*x+4).deriv(x))
    a = (3 * x ** 5 + 4 * x + 12).deriv(x)
    print(a, '(x=0.5)=', a.eval(x=0.5))

    b = e ** (-3 * x ** 2 - 3 * x + 4)
    a = b.deriv(x)
    print(b)
    print(a)
    print((x ** 0.5).deriv(x))

    print(sin(math.pi / 2))
    print((x ** 2 + y ** 2).deriv(y).eval(y=0.5))
    print()


if __name__ == '__main__':
    main()