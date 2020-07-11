# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p52_neg.py
@Description    :  
@CreateTime     :  2020/7/6 09:31
------------------------------------
@ModifyTime     :  
"""


class Complex:
    def __init__(self, real, virtual):
        self.real = real
        self.virtual = virtual

    def __add__(self, other):
        return Complex(self.real + other.real, self.virtual + other.virtual)

    def __sub__(self, other):
        return Complex(self.real - other.real, self.virtual - other.virtual)

    def __mul__(self, other):
        real = self.real * other.real - self.virtual * other.virtual
        virtual = self.virtual * other.real + self.real * other.virtual
        return Complex(real, virtual)

    def __truediv__(self, other):
        m = other.real ** 2 + other.virtual ** 2
        real = self.real * other.real + self.virtual * other.virtual
        virtual = - self.virtual * other.real + self.real * other.virtual
        return Complex(real/m, -virtual/m)

    def __repr__(self):
        return '%s + %si' % (self.real, self.virtual)


def main():
    c1 = Complex(3, 4)
    c2 = Complex(2, -1)

    print('c1 + c2', c1 + c2)
    print('c1 - c2', c1 - c2)
    print('c1 * c2', c1 * c2)
    print('c1 / c2', c1 / c2)


if __name__ == '__main__':
    main()