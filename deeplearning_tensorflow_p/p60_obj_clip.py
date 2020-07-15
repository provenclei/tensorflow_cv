# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :
------------------------------------
@File           :  p60_obj_clip.py
@Description    :
@CreateTime     :  2020/7/13 11:39
------------------------------------
@ModifyTime     : 图像裁剪，图像语义分割
"""
import numpy as np


def obj_clip(img, foreground, border=None):
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
                    [x1, y1], [x2, y2], ... # 第二个物体中横纵坐标
                   ],
                ......
            ]
    '''
    result = []
    height, width = np.shape(img)
    visited = set()
    for h in range(height):
        for w in range(width):
            if img[h, w] == foreground and not (h, w) in visited:
                obj = visit(img, height, width, h, w, visited, foreground, border)
                result.append(obj)
    return result


def visit(img, height, width, h, w, visited, foreground, border):
    visited.add((h, w))
    result = [(h, w)]

    if w > 0 and not (h, w-1) in visited:
        if img[h, w-1] == foreground:
            result += visit(img, height, width, h, w-1, visited, foreground, border)
        elif border is not None and img[h, w-1] == border:
            # 将边界划分在每个类别中
            result.append((h, w-1))

    if w < width - 1 and not (h, w+1) in visited:
        if img[h, w+1] == foreground:
            result += visit(img, height, width, h, w+1, visited, foreground, border)
        elif border is not None and img[h, w+1] == border:
            result.append((h, w+1))

    if h > 0 and not (h-1, w) in visited:
        if img[h-1, w] == foreground:
            result += visit(img, height, width, h-1, w, visited, foreground, border)
        elif border is not None and img[h-1, w] == border:
            result.append((h-1, w))

    if h < height - 1 and not (h+1, w) in visited:
        if img[h+1, w] == foreground:
            result += visit(img, height, width, h+1, w, visited, foreground, border)
        elif border is not None and img[h+1, w] == border:
            result.append((h+1, w))
    return result


def main():
    import sys
    sys.setrecursionlimit(9000000)

    img = np.zeros([400, 400])
    import cv2
    cv2.rectangle(img, (10, 10), (150, 150), 1.0, 1)
    cv2.circle(img, (270, 270), 70, 1.0, 10)
    cv2.line(img, (100, 10), (100, 150), 0.5, 10)
    cv2.putText(img, 'ABC', (200, 200), cv2.FONT_HERSHEY_PLAIN, 2.0, 1.0, 2)

    cv2.imshow('my image', img)
    cv2.waitKey()

    print(img)
    objs = obj_clip(img, 1.0, 0.5)
    for obj in objs:
        clip = np.zeros([400, 400])
        for h, w in obj:
            clip[h, w] = 0.5
        cv2.imshow('my image', clip)
        cv2.waitKey()


if __name__ == '__main__':
    main()
