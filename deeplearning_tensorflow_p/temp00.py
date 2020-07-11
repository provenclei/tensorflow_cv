# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  temp00.py
@Description    :  
@CreateTime     :  2020/6/24 09:11
------------------------------------
@ModifyTime     :

首先在系统缺省目录下寻找
然后在当前项目目录下寻找


import 规则：
1. 主程序下不能使用相对路径
2. 子程序下可使用相对路径
3. 相对路径只能通过 from 引用  （？？？）
4. pycharm 可以指定任意 source route
    make directory as 属性

"""
# import temp_sub_dir.temp01 as temp01

# from .temp_sub_dir import temp01
# temp01.a()


import temp_sub_dir.temp02 as temp02

