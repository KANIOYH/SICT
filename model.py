#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件    :model.py\n
@说明    :模型层，集成接口\n
@时间    :2023/03/12 22:11:22\n
@作者    :陈源辉\n
@版本    :1.0\n
'''


import numpy as np
import frequency_dct as dct
import cv2
import SIFT as sift
import hamming as hm
import socket
import datetime
import fileio


emb_info = []
ext_info = []


def diff(wm1, wm2):
    """计算两张水印的diff值
    
    基于异或

    Parameters
    ----------
    wm1:np.ndarray
        水印序列1
    wm2:np.ndarray
        水印序列2

    Returns
    ----------
    int
        wm1与wm2的异或和
    """
    wm1 = np.bool8(wm1)
    wm2 = np.bool8(wm2)
    res = wm1 ^ wm2
    return res.sum()


def twoGroupsFilter(group1, group2, th):
    """水印筛选方法
    
    对两个水印组的水印进行筛选

    Parameters
    ----------
    group1:list
        水印组1
    group2:list
        水印组2
    th:int
        两张水印的所允许差异值的上限

    Returns
    ----------
    list
        从两组水印中提取出的差异最小的一对水印
    """
    res = []
    min = 1e18
    for wm1 in group1:
        for wm2 in group2:
            differ = diff(wm1, wm2)
            if differ <= th and differ < min:
                min = differ
                res = []
                res.append(wm1)
                res.append(wm2)
    return res


def watermarkFilter(groups, th):
    """水印分组筛选
    
    对所有水印组进行两两筛选

    Parameters
    ----------
    groups:list
        初步获得的全部水印组
    th:int
        两张水印的所允许差异值的上限

    Returns
    ----------
    list
        所有经过筛选获得的水印集合（即有效水印组）
    """
    res = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            res.extend(twoGroupsFilter(groups[i], groups[j], th))
    return res


def finalWatermark(wList, msgShape):
    """水印提纯方法
    
    对有效水印组中的最多k*(k-1)张水印进行统计，合成最终的一张水印

    Parameters
    ----------
    wList:list
        经过筛选后所获得的有效水印组
    msgShape:tuple(int, int)
        水印矩阵的长款

    Returns
    ----------
    list
        最终提取出的水印序列（二进制列表）
    """
    a, b = msgShape
    siz = a * b
    watermark = np.zeros((a, b))
    tmp = np.zeros(siz, np.int)
    count = 0
    for wm in wList:
        if sum(wm) < len(wm) * 0.12 or sum(wm) > len(wm) * 0.88:
            continue
        rNums = (len(wm) - 1) % 14 + 1
        if np.sum(wm[len(wm) - rNums:]) >= 0.5 * rNums:
            wm = np.array(wm) ^ 1
            wm = list(wm)
        if len(wm) == siz:
            tmp = tmp + wm
            count += 1
    tmp = np.reshape(tmp, (a, b))
    print(tmp)
    for x in range(a):
        for y in range(b):
            if tmp[x][y] >= count / 2:
                watermark[x][y] = 1
            else:
                watermark[x][y] = 0
    return np.uint8(watermark)


def globalExtract(img_path, msg_shape, k, th):
    """提取水印的控制方法
    
    注意默认参数

    Parameters
    ----------
    img_path: str
        待提取水印图片的路径
    msgShape:tuple(int, int)
        水印矩阵的长款
    k:int
        待嵌入区域的个数
    th:int
        提取阈值

    Returns
    ----------
    tuple()
        最终提取出的水印信息(各部分文字信息构成的元组)
    """
    width, height, bands, dct_img, geotrans, proj = fileio.readTif(img_path)
    # em_img = cv2.imread(img_path)
    a, b = msg_shape

    # em_ycbcr = cv2.cvtColor(em_img, cv2.COLOR_BGR2YCrCb)
    # dct_img = em_ycbcr[..., 0]

    wm_list = sift.getKeyPoints(dct_img, msg_shape, k * 2)
    wm_positions = dct.transToCoordinates(wm_list)
    wm_blocks = dct.get_block(dct_img, msg_shape, wm_positions)
    rst = dct.extract(dct_img, wm_blocks)

    wm = watermarkFilter(rst, th)
    wm = finalWatermark(wm, (a, b))
    wm = np.ndarray.flatten(wm)

    info = hm.decode(wm)
    return info


def watermarking(msg_ext, msg_shape):
    """水印生成方法
    
    处理所输入水印信息的方法

    Parameters
    ----------
    msg_ext: str
        用户自定义的额外的水印信息
    msgShape:tuple(int, int)
        水印矩阵的长宽

    Returns
    ----------
    np.ndarray
        经过处理后得到的水印消息二进制矩阵
    """
    a, b = msg_shape
    msg_base = get_base_msg()
    msg = hm.encode(msg_base, msg_ext)

    if len(msg) + 1 > a * b:
        msg = msg[:int(int((a * b - 1) / 14) * 14)]

    while len(msg) < a * b:
        msg.append(0)
    msg = np.reshape(msg[:a * b], (a, b))
    return msg


def screenEmbed(img_buffer, msg_ext, msg_shape, k, r):
    """屏幕嵌入方法
    
    获取所有待嵌入区域的坐标和嵌入后的图像块

    Parameters
    ----------
    img_buffer: np.ndarray
        当前显示器的屏幕图像矩阵
    msg_ext: str
        用户自定义的额外水印信息
    msgShape: tuple(int, int)
        水印矩阵的长宽
    k: int
        待嵌入区域的个数
    r: float
        嵌入强度系数

    Returns
    ----------
    list
        已完成嵌入的图像块
    """

    ycbcr = cv2.cvtColor(img_buffer, cv2.COLOR_BGR2YCrCb)
    img = ycbcr[..., 0]

    lst = sift.getKeyPoints(img, msg_shape, k)
    positions = dct.transToCoordinates(lst)
    blocks = dct.get_block(img, msg_shape, positions)

    dct_img_dict = dct.insert_get(ycbcr, blocks, msg_ext.flatten(), r)
    return dct_img_dict


def get_base_msg():
    """
    获取基本系统信息(当前IP、系统时间)\n

    Parameters
    ----------

    Returns
    ----------
    str
        当前IP、时间构成的字符串
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except BaseException:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except BaseException:
            print("NOT FIND NET")
            ip = "000.000.000.000"
    finally:
        s.close()

    a = datetime.datetime.now()

    def format_Out(data, n):
        data = str(data)
        while len(data) < n:
            data = "0" + data
        if len(data) > n:
            data = data[n:]
        return data

    ip = str(ip).split(".")
    for i in range(len(ip)):
        ip[i] = format_Out(ip[i], 3)
    ip = ''.join(ip)
    res = ip + format_Out(a.year, 2) + format_Out(a.month, 2) + format_Out(a.day, 2) + format_Out(a.hour,
          2) + format_Out(a.minute, 2)
    return compress_base_msg(res)


def compress_base_msg(msg_base):
    """时间信息压缩
    
    将时间信息压缩到一个32位整数上，用以节省水印空间\n

    Parameters
    ----------
    msg_base: str
        基本信息构成的01字符串

    Returns
    ----------
    tuple
        当前IP、时间构成的列表
    """
    tmp=list(msg_base)
    tmp[12],tmp[13],tmp[14],tmp[15] = tmp[14],tmp[15],tmp[12],tmp[13]
    base_msg = "".join(tmp)
    msg_ip=".".join([base_msg[i:i+3] for i in range(0,12,3)])
    msg_time=int("".join(tmp[12:22]))
    return (msg_ip, msg_time)


def globalEmbed(img_path, out_path, msg_ext, r=1, k=8, msg_shape=(12, 12)):
    a, b = msg_shape
    # 设置校验位
    # print(msg_ip,msg_ext)
    msg_base = get_base_msg()
    msg = hm.encode(msg_base, msg_ext)

    print(msg)
    global emb_info
    emb_info = msg

    # print(len(msg))
    # 信息太长！
    if len(msg) + 1 > a * b:
        msg = msg[:int(int((a * b - 1) / 14) * 14)]
        emb_info = msg

    # 抵抗反水印的校验位默认设为0
    while len(msg) < a * b:
        msg.append(0)
    msg = np.reshape(msg[:a * b], (a, b))
    # k为嵌入区域的个数
    a = len(msg)
    b = len(msg[0])
    # msg_shape: 水印矩阵的尺寸
    msg_shape = (a, b)
    
    #ori_img = np.array(fileio.readTif(img_path))
    #ori_img = np.array(Image.open(img_path))
    width, height, bands, data, geotrans, proj = fileio.readTif(img_path)
    img = data
    print("img ", type(img),img.shape)
    # # 转化为ycbcr图
    # ycbcr = cv2.cvtColor(ori_img, cv2.COLOR_BGR2YCrCb)
    # # img为y通道的分量
    # img = ycbcr[..., 0]

    # 获取关键点
    lst = sift.getKeyPoints(img, msg_shape, k)
    positions = dct.transToCoordinates(lst)
    blocks = dct.get_block(img, msg_shape, positions)

    # 水印嵌入
    # dct_img = dct.insert(img, blocks, msg.flatten(), r)
    # data = dct_img
    # embeded = data
    embeded = dct.insert(img, blocks, msg.flatten(), r)
    fileio.writeTiff(embeded, geotrans, proj, out_path)
    # embeded = np.copy(ycbcr)
    # embeded[..., 0] = np.uint8(dct_img)
    # embeded = cv2.cvtColor(embeded, cv2.COLOR_YCrCb2BGR)
    # imageio.imsave(out_path, embeded)
    return width, height, bands