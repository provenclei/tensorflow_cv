# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p68_mutilstm.py
@Description    :  
@CreateTime     :  2020/7/25 21:10
------------------------------------
@ModifyTime     :  多层 lstm
                每个 cell 都应该具有 zero_state 和 call 函数
"""
import tensorflow as tf


class MyMultiRNNCell:
    def __init__(self, cells):
        self.cells = cells

    def zero_state(self, batch_size, dtype):
        return [cell.zero_state(batch_size, dtype) for cell in self.cells]

    def __call__(self, inputs, states):
        new_states = []
        for cell, state in zip(self.cells, states):
            inputs, new_state = cell(inputs, state)
            new_states.append(new_state)
        return inputs, new_states