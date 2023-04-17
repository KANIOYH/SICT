#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件    :SIFT.py\n
@说明    :特征区域获取模块\n
@时间    :2022/03/12 22:13:46\n
@作者    : SICT 制作委员会\n
@版本    :1.0\n
'''


from operator import attrgetter
import cv2
import numpy
import numpy as np
from numba import jit



def checkIntersect(mp, point, rHeight, rWidth):
    """判断特征区域是否重叠方法
    
    基于矩形是否叠置思想

    Parameters
    ----------
    mp:np.ndarray
        辅助数组
    point:tuple(int,int)
        特征区域中心点坐标
    rHeight:int
        特征区域长度的一半
    rWidth:int
        特征区域宽度的一半

    Returns
    ----------
    bool
        是否重叠
    """
    cx, cy = point
    top = round(cx - rHeight)
    bottom = round(cx + rHeight)
    left = round(cy - rWidth)
    right = round(cy + rWidth)

    count = np.sum(mp[top:top + 1, left:right]) + np.sum(mp[bottom - 1:bottom, left:right]) + \
            np.sum(mp[top:bottom, left:left + 1]) + np.sum(mp[top:bottom, right - 1:right])
    if count > 0:
        return False
    return True


def updateRectangle(mp, point, rHeight, rWidth):
    """更新辅助数组方法
    
    更新辅助数组方法

    Parameters
    ----------
    mp:np.ndarray
        辅助数组
    point:tuple(int,int)
        特征区域中心点坐标
    rHeight:int
        特征区域长度的一半
    rWidth:int
        特征区域宽度的一半

    Returns
    ----------
    np.ndarray
        更新后的辅助数组
    """
    cx, cy = point
    top = round(cx - rHeight)
    bottom = round(cx + rHeight)
    left = round(cy - rWidth)
    right = round(cy + rWidth)

    mp[top:top + 1, left:right] = True
    mp[bottom - 1:bottom, left:right] = True
    mp[top:bottom, left:left + 1] = True
    mp[top:bottom, right - 1:right] = True
    return mp


# 判断纯色区域
def judgeDocArea(block):
    """判断纯色区域
    
    注意阈值

    Parameters
    ----------
    block:np.ndarray
        特征区域图像块

    Returns
    ----------
    bool
        是否为纯色区块
    """
    area = len(block) * len(block[0])
    block = np.ndarray.flatten(block)
    block = numpy.bincount(block)
    count = block[220:].sum()

    if count / area > 0.9:
        return False
    return True


def getKeyPoints(img, msgShape, k):
    """获取关键点的序列
    
    注意参数输入

    Parameters
    ----------
    img:np.ndarray
        待嵌入图像
    msgShape:tuple(int,int)
        水印矩阵的长与宽
    k:int
        待嵌入区域的个数

    Returns
    ----------
    list(cv2.KeyPoint())
        候选关键点序列
    """
    mp = np.zeros((len(img), len(img[0])), np.bool8)
    count = 0
    max_count = 0
    a, b = msgShape
    rHeight = a * 4
    rWidth = b * 4

    res = []
    sift = cv2.SIFT_create()
    kps = list(sift.detect(img, None))
    kps.sort(key=attrgetter('size'), reverse=True)

    for point in kps:
        max_count += 1
        if max_count > k * 3:
            break
        x = round(point.pt[1])
        y = round(point.pt[0])
        if x >= rHeight and x + rHeight <= len(img) and y >= rWidth and y + rWidth <= len(img[0]):
            judge1 = checkIntersect(mp, (x, y), rHeight, rWidth)
            if judge1 == True:
                judge2 = judgeDocArea(img[x - rHeight:x + rHeight, y - rWidth:y + rWidth])
                # judge2 = True
                if judge2 == True:
                    mp = updateRectangle(mp, (x, y), rHeight, rWidth)
                    count += 1
                    res.append(point)
        if count >= k:
            break
    return res
