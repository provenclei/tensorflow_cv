# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t52_Complex.py
@Description    :  
@CreateTime     :  2020/7/6 13:44
------------------------------------
@ModifyTime     :  实现负数运算
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
        return Complex(self.real * other.real - self.virtual * other.virtual,
                       self.real * other.virtual + self.virtual * other.real)

    def __truediv__(self, other):
        derive = other.real ** 2 + other.virtual ** 2
        return Complex((self.real * other.real + self.virtual * other.virtual)/derive,
                       (-self.real * other.virtual + self.virtual * other.real)/derive)

    def __repr__(self):
        return '%s + %si' % (self.real, self.virtual)


def main():
    c1 = Complex(3, 4)
    c2 = Complex(2, -1)

    print('c1 + c2 = ', c1 + c2)
    print('c1 - c2 = ', c1 - c2)
    print('c1 * c2 = ', c1 * c2)
    print('c1 / c2 = ', c1 / c2)


if __name__ == '__main__':
    main()