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
import sys

sys.setrecursionlimit(9000000)


def obj_clip(img, foreground, background, border):
    '''
    在已知前景背景和边界的情况下对实例进行分割
    :param img: np.ndarray
    :param foreground: 前景 1
    :param background: 背景 0
    :param border: 边界 2
    :return: [
                [
                    [x1, y1], [x2, y2], ...  # 第一个物体中横纵坐标
                    ],
                [
                    [x1, y1], [x2, y2], ...], # 第二个物体中横纵坐标
                   ],
                ......
            ]
    '''
    height, width = np.shape(img)
    # {(h,w):k}, k：第几个实体，0：未访问过
    visited = {}
    flag = 0
    for h in range(height):
        for w in range(width):
            if img[h, w] == foreground and not visited.get((h, w)):
                results = visit(img, h, w, foreground, background, border)
                # 将访问过的点加入到 visited 中
                for result in results:
                    visited[result] = flag
                    flag += 1
    print(visited)


def visit(img, h, w, foreground, background, border):
    results = set()
    tup = round(img, h, w, foreground, background, border)
    results.add(tup)
    return results


def round(img, h, w, foreground, background, border):
    '''
    访问所有边界内所有前景相邻的点
    :param img:
    :param h:
    :param w:
    :return: 所有点的集合
    '''
    if img[h, w] == border or img[h, w] == background:
        return ()
    # 选择上下左右的点
    # (h+1, w), (h-1, w), (h, w+1), (h, w-1)
    visit(img, h-1, w, foreground, background, border)
    visit(img, h+1, w, foreground, background, border)
    visit(img, h, w-1, foreground, background, border)
    visit(img, h, w+1, foreground, background, border)
    return tuple(h, w)


def main():
    test_arr = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 2, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 2, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 2, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 2, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    obj_clip(test_arr, 1, 0, 2)


if __name__ == '__main__':
    main()