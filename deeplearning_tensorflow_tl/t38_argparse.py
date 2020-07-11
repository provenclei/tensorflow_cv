# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p36_argparse.py
@Description    :  
@CreateTime     :  2020/6/22 19:19
------------------------------------
@ModifyTime     :  
"""
import argparse


def main():
    # ArgumentParser 将命令行解析为 python 对象
    parser = argparse.ArgumentParser()
    # 向 ArgumentParser 对象中添加命令行中的参数
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    # parser.add_argument('--new_model', type=bool, default=False, help='whether using new model')
    # 相当于一个开关，运行时会触发 store 的值
    parser.add_argument('--new_model', default=False, help='whether using new model', action='store_true')

    # parse_args：命令行参数解析方法
    a = parser.parse_args()
    print(a.batch_size)
    print(a.lr)
    print(a.new_model)


if __name__ == '__main__':
    main()