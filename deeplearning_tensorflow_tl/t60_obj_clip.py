# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t60_obj_clip.py
@Description    :  
@CreateTime     :  2020/7/14 14:04
------------------------------------
@ModifyTime     :  图像裁剪，图像语义分割
"""
import numpy as np


def obj_clip(img, foreground, border=None):
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
    if w > 0 and not (h, w - 1) in visited:
        if img[h, w-1] == foreground:
            result += visit(img, height, width, h, w - 1, visited, foreground, border)
        elif border is not None and img[h, w - 1] == border:
            result.append((h, w - 1))

    if w < width - 1 and not (h, w + 1) in visited:
        if img[h, w + 1] == foreground:
            result += visit(img, height, width, h, w + 1, visited, foreground, border)
        elif border is not None and img[h, w + 1] == border:
            result.append((h, w + 1))

    if h > 0 and not (h - 1, w) in visited:
        if img[h - 1, w] == foreground:
            result += visit(img, height, width, h - 1, w, visited, foreground, border)
        elif border is not None and img[h - 1, w] == border:
            result.append((h - 1, w))

    if h < height - 1 and not (h + 1, w) in visited:
        if img[h + 1, w] == foreground:
            result += visit(img, height, width, h + 1, w, visited, foreground, border)
        elif border is not None and img[h + 1, w] == border:
            result.append((h + 1, w))
    return result


def main():
    import cv2
    import sys

    img = np.zeros([400, 400])

    sys.setrecursionlimit(200000)
    cv2.rectangle(img, (10, 10), (150, 150), 1, 5)
    cv2.circle(img, (270, 270), 70, 1, 10)
    cv2.line(img, (100, 10), (100, 250), 0.5, 10)
    cv2.putText(img, 'zoo', (200, 200), cv2.FONT_HERSHEY_PLAIN, 2, 1, 2)
    # cv2.arrowedLine(img, (100, 100), (150, 150), 1)

    cv2.imshow('my pic', img)
    cv2.waitKey()

    objs = obj_clip(img, 1, 0.5)
    for obj in objs:
        clip = np.zeros([400, 400])
        for h, w in obj:
            clip[h, w] = 0.5
        cv2.imshow('my image', clip)
        cv2.waitKey()


if __name__ == '__main__':
    main()