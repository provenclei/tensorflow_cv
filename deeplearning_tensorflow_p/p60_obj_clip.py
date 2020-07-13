# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p60_obj_clip.py
@Description    :  
@CreateTime     :  2020/7/13 11:39
------------------------------------
@ModifyTime     :  
"""
import numpy as np


def obj_clip(img, foreground, background, border):
    '''

    :param img: np.ndarray
    :param foreground:
    :param background:
    :param border:
    :return: [
                [
                    [x1, y1], [x2, y2], ...]],
                [
                    [x1, y1], [x2, y2], ...],
                   ...
                ]
                ]
    '''
    height, width = np.shape(img)
    visited = {}
    for h in range(height):
        for w in range(width):
            if img[h, w] == foreground and not visited[(h, w)]:
                visit(img, h, w)



def main():
    pass


if __name__ == '__main__':
    main()