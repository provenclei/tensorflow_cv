# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  temp.py
@Description    :  
@CreateTime     :  2020/6/22 15:27
------------------------------------
@ModifyTime     :  
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--new_model', default=False, help='new model', action='store_true')

    a = parser.parse_args()
    print(a.lr)
    print(a.batch_size)
    print(a.new_model)


if __name__ == '__main__':
    main()