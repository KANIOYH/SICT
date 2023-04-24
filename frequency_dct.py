#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件    :frequency_dct.py
@说明    :dct中频系数交换模块
@时间    :2023/03/11 10:12:58
@作者    : 陈源辉
@版本    :1.0
'''



import numpy as np
import cv2
import random
from numba import jit

"""
    参考灰亮度量化表
    ----------
        [16, 11, 10, 16, 24, 40, 51, 61 ], \n
        [12, 12, 14, 19, 26, 58, 60, 55 ], \n
        [14, 13, 16, 24, 40, 57, 69, 56 ], \n
        [14, 17, 22, 29, 51, 87, 80, 62 ], \n
        [18, 22, 37, 56, 68, 109,103,77 ], \n
        [24, 35, 55, 64, 81, 104,113,92 ], \n
        [49, 64, 78, 87, 103,121,120,101], \n
        [72, 92, 95, 98, 112,100,103,99 ]  \n
"""


q1 = 51 
"""
变量
    8*8块dct中频系数值 q1 坐标(3,4) 
"""

q2 = 56
"""
变量
    8*8块dct中频系数值 q2 坐标(4,3)
"""


def get_block(img:np.ndarray, msg_shape:tuple, positions:list or tuple):
    '''图像获取特征区域
    
    根据 msg_shape 大小，给特征位置分配 img 中的图像区域,特征区域大小为a*b*8*8的块\n
    注意！由于长宽乘上8后必是偶数，因此特征值无法在特征区域的中心。中心在目标区域中心四格的右下位
         
    Parameters
    ----------
    img:np.ndarray
        希望获得特征区域的图像
    msg_shape:tuple(int,int)
        信息水印长宽
    positions:list
        特征点位置集合
        
    Returns
    ----------
    特征区域组，存储每个特征区域的对角线坐标
    '''
    blocks = []
    # 此处有修改
    long = int(msg_shape[0] * 4)
    width = int(msg_shape[1] * 4)
    for i in positions:
        blocks.append([i[0] - long, i[1] - width, i[0] + long, i[1] + width])
    return np.array(blocks)


@jit
def preProcess(block:np.ndarray):
    """预处理，对纯色区块添加随机底纹
    
    采取对区域各栅格减去一个小范围随机数的方法。\n
    注意！此处采取了jit加速，第一次运行会先编译方法耗时较大
    
    Parameters
    ----------
    block:np.ndarray
        RGB格式的图像区域数据
        
    Returns
    ----------
    np.ndarray
        修改后的数据
    """
    posb = []
    posw = []
    for i in range(len(block)):
        for j in range(len(block[i])):
            if block[i][j] > 245:
                posw.append((i, j))
            elif block[i][j] < 10:
                posb.append((i, j))

    for pixel in posw:
        x, y = pixel
        block[x][y] -= random.randint(6, 7)
    for pixel in posb:
        x, y = pixel
        block[x][y] += random.randint(8, 10)
    return block


@jit
def afterProcess(pre_img:np.ndarray, after_img:np.ndarray):
    """后处理，对像素点插值以消除噪声
    
    对处理前后图像进行比较，修改变换过大区域\n
    注意！此处采取了jit加速，第一次运行会先编译方法耗时较大
    
    Parameters
    ----------
    pre_img:np.ndarray
        嵌入前区域
    fter_img:np.ndarray
        嵌入后区域
        
    Returns
    ----------
    np.ndarray
        修改后的数据
    """
    alpha = 0.5
    for i in range(len(after_img)):
        for j in range(len(after_img[i])):
            # after_img[i][j] = np.uint8(0.65 * pre_img[i][j] + 0.35 * after_img[i][j])
            if abs(after_img[i][j] - pre_img[i][j]) > 160:
                after_img[i][j] = pre_img[i][j]
            elif after_img[i][j] < 30 or after_img[i][j] > 225 :
                after_img[i][j] = (1 - alpha) * pre_img[i][j] + alpha * after_img[i][j]
            # after_img[i][j] = np.uint8(after_img[i][j])
    return after_img


def insert(img:np.ndarray, blocks:list, msg:list, r:int):
    """对特征区域嵌入水印方法
    
    将图像进行dct变换后，根据 msg 信息修改 C1 , C2 的值，达到嵌入的目的
    
    Parameters
    ----------
    img:np.ndarray
        待嵌入图像
    blocks:list
        特征区域的对角线坐标
    msg:list
        经过BCH编码的消息队列，只包含0,1的 int type
    r:int
        嵌入强度系数
        
    Returns
    ----------
    np.ndarray
        嵌入后的图像
    """
    Cx1 = 3
    Cy1 = 4
    Cx2 = 4
    Cy2 = 3
    rr = 45 * r
    # 一个特征区域的8*8的单元组
    for block in blocks:
        top, left, bottom, right = block
        index = 0  # msg bit位指针

        # 预处理
        pre_img = np.array(img[top:bottom, left:right], np.int)
        after_img = preProcess(np.copy(img[top:bottom, left:right]))

        for i in range(0, len(after_img), 8):
            for j in range(0, len(after_img[i]), 8):

                temp = cv2.dct(np.float32(after_img[i:i + 8, j:j + 8]))
                C1 = temp[Cx1, Cy1]
                C2 = temp[Cx2, Cy2]
                if msg[index] == 0:
                    dx = (rr - temp[Cx1, Cy1] + temp[Cx2, Cy2]) / 2
                    if dx > 0:
                        C1 = temp[Cx1, Cy1] + dx
                        C2 = temp[Cx2, Cy2] - dx
                else:
                    dx = (rr - temp[Cx2, Cy2] + temp[Cx1, Cy1]) / 2
                    if dx > 0:
                        C1 = temp[Cx1, Cy1] - dx
                        C2 = temp[Cx2, Cy2] + dx
                temp[Cx1, Cy1] = np.float32(C1)
                temp[Cx2, Cy2] = np.float32(C2)
                after_img[i:i + 8, j:j + 8] = np.uint8(cv2.idct(temp))
                index += 1

        # 后处理
        after_img = np.array(after_img, np.int)
        after_img = afterProcess(pre_img, after_img)
        img[top:bottom, left:right] = after_img

    return img


def insert_get(ycbcr:np.ndarray, blocks:list, msg:list, r=1):
    """对特征区域嵌入水印方法，只返回特征区域位置和数据
    
    将图像进行dct变换后，根据 msg 信息修改 C1 , C2 的值，达到嵌入的目的
    
    Parameters
    ----------
    ycbcr:np.ndarray
        待嵌入图像 YCbCr格式
    blocks:list
        特征区域的对角线坐标
    msg:list
        经过BCH编码的消息队列，只包含0,1的 int type
    r:int
        嵌入强度系数,默认r=1
        
    Returns
    ----------
    dict
        以坐标为健，数据为值的字典
    """


    Cx1 = 3
    Cy1 = 4
    Cx2 = 4
    Cy2 = 3
    rr = 45 * r
    img = ycbcr[..., 0]
    result_dict = {}
    for block in blocks:
        top, left, bottom, right = block
        long = block[2] - block[0]
        width = block[3] - block[1]
        index = 0  # msg bit位指针

        # 预处理
        pre_img = np.array(img[top:bottom, left:right], np.int)
        after_img = preProcess(np.copy(img[top:bottom, left:right]))

        for i in range(0, len(after_img), 8):
            for j in range(0, len(after_img[i]), 8):
                temp = cv2.dct(np.float32(after_img[i:i + 8, j:j + 8]))
                C1 = temp[Cx1, Cy1]
                C2 = temp[Cx2, Cy2]
                if msg[index] == 0:
                    dx = (rr - temp[Cx1, Cy1] + temp[Cx2, Cy2]) / 2
                    if dx > 0:
                        C1 = temp[Cx1, Cy1] + dx
                        C2 = temp[Cx2, Cy2] - dx
                else:
                    dx = (rr - temp[Cx2, Cy2] + temp[Cx1, Cy1]) / 2
                    if dx > 0:
                        C1 = temp[Cx1, Cy1] - dx
                        C2 = temp[Cx2, Cy2] + dx
                temp[Cx1, Cy1] = np.float32(C1)
                temp[Cx2, Cy2] = np.float32(C2)
                after_img[i:i + 8, j:j + 8] = np.uint8(cv2.idct(temp))
                index += 1

        # 后处理
        after_img = np.array(after_img, np.int)
        after_img = afterProcess(pre_img, after_img)
        img[top:bottom, left:right] = after_img

        i = block[0]
        j = block[1]
        ycbcr_part = np.dstack((
            img[i:i + long, j:j + width],
            ycbcr[i:i + long, j:j + width, 1],
            ycbcr[i:i + long, j:j + width, 2]))
        result_dict[(i, j)] = cv2.cvtColor(ycbcr_part, cv2.COLOR_YCrCb2BGR)
    return result_dict


def extract(img:np.ndarray, blocks:list):
    """提取水印的方法
    
    blocks有2k个特征区域，函数更改为返回2k*9个提取的水印信息
    
    Parameters
    ----------
    img:np.ndarray
        待溯源图像
    blocks:list
        特征区域组，存储每个特征区域的对角线坐标

        
    Returns
    ----------
    list
        水印消息的 bit（0/1） numpy数组。有多组，理论上每组应该一样，可作为后期校验与修正的依据
    """

    # 周围17个像素点
    # dx = [-1, -1, 0, 0, 0, 1, 1, -2, -2, -2, -2, 2, 2, 2, 2, -1, -1, 1, 1]
    # dy = [-1, 0, -1, 0, 1, 0, 1, -2, -1, 1, 2, -2, -1, 1, 2, -2, 2, -2, 2]

    # 周围25个像素点
    dx = [-2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    dy = [-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]

    msg = []

    Cx1 = 3
    Cy1 = 4
    Cx2 = 4
    Cy2 = 3

    height = len(img)
    width = len(img[0])

    for block in blocks:
        # 每个特征区域包含9个水印图像（打包为一组）
        group_msg = []
        top, left, bottom, right = block
        # 遍历一个特征区域的8*8的单元组
        for p in range(len(dx)):
            # 边界检查
            if top + dx[p] < 0 or bottom + dx[p] > height - 1 or left + dy[p] < 0 or right + dy[p] > width - 1:
                continue

            now_msg = []
            count = 0
            for i in range(top + dx[p], bottom + dx[p], 8):
                for j in range(left + dy[p], right + dy[p], 8):
                    temp = cv2.dct(np.float32(img[i:i + 8, j:j + 8]))
                    if temp[Cx1, Cy1] > temp[Cx2, Cy2]:
                        now_msg.append(0)
                    else:
                        now_msg.append(1)
            group_msg.append(now_msg)
        msg.append(group_msg)
    # 返回一个 (2*k)*(9)*(a*b)大小的水印组
    return np.array(msg)

def transToCoordinates(keyPoints):
    """转化为二维坐标方法
    
    cv2.KeyPoint类型转化为二维坐标列表
    
    Parameters
    ----------
    keyPoints
        cv2.KeyPoint类型
        
    Returns
    ----------
    list
        二维坐标列表
    """
    kps = []
    for pt in keyPoints:
        x = round(pt.pt[1])
        y = round(pt.pt[0])
        kps.append((x, y))
    return kps
