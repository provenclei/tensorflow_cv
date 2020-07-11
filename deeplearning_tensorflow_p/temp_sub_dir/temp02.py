# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  temp02.py
@Description    :  
@CreateTime     :  2020/6/24 09:15
------------------------------------
@ModifyTime     :

如果当前目录是主程序，则不能使用相对目录
即：主程序不能使用相对目录


"""
import os

print(' in temp02')


# from TF_turial.artifitial_intelligent_p import temp_sub_dir
# import temp_sub_dir.temp01 as temp01

# from TF_turial.artifitial_intelligent_p.temp_sub_dir import temp01 as temp01

from . import temp01


def main():
    pass


if __name__ == '__main__':
    main()