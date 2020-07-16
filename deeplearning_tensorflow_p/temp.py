# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  temp.py
@Description    :  
@CreateTime     :  2020/6/23 09:34
------------------------------------
@ModifyTime     :
VAE: 变分自编码器（无监督模型）
显示分布
encoder -> logits/sentiment logits（语义） -> decoder
            正态分布
通常使用绝对值误差或均方差误差
生成数字图片，logits长度为4
生成人脸，logits长度为100

ubuntu 中常用命令：
watch -n 1 nvidia-smi  每一秒更新一次 GPU 的使用情况
tail -f nohup.out  查看窗口的更新情况

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
                results = visit(img, height, width, h, w, foreground, background, border)
                # 将访问过的点加入到 visited 中
                for result in results:
                    visited[result] = flag
                    flag += 1
    print(visited)


def visit(img,height, width, h, w, foreground, background, border):
    results = set()
    tup = round(img, height, width, h, w, foreground, background, border)
    results.add(tup)
    return results


def round(img, height, width, h, w, foreground, background, border):
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
    if h > 0:
        round(img, height, width, h-1, w, foreground, background, border)
        return tuple((h-1, w))
    if h < height - 1:
        round(img, height, width, h+1, w, foreground, background, border)
        return tuple((h+1, w))
    if w > 0:
        round(img, height, width, h, w-1, foreground, background, border)
        return tuple((h, w-1))
    if w < width - 1:
        round(img, height, width, h, w+1, foreground, background, border)
        return tuple((h, w+1))


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