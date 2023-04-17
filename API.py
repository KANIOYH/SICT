#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件    :SICT_API.py\n
@说明    :对外调用API接口模块,将功能整合到一起\n
@时间    :2023/03/12 21:57:24\n
@作者    : 陈源辉\n
@版本    :1.0\n
'''


import model

# 水印嵌入
def embed(img_path:str, out_path:str, msg:str, r=1.6, k=10, msg_shape=(16, 16)):
    """单纯对于图像的嵌入方法
    
    注意嵌入参数
    
    Parameters
    ----------
    img_path:string
        需要嵌入的图像URL地址
    out_path:string
        保存的路径与名称URL
    msg:string
        希望写入的自定义信息
    k:int
        嵌入特征区域的个数，默认k=12
    r:float
        水印嵌入强度，r越大水印嵌入越强，同时也会降低图像质量。默认：r=1.60
        
    Returns
    ----------
    NULL
    """
    return model.globalEmbed(img_path, out_path, msg, r, k, msg_shape)


# 水印提取
def extract(img_path, msg_shape=(16, 16), k=10, th=15):
    """对于图像的进行溯源的方法
    
    溯源参数需要与嵌入时保持一致。
    
    Parameters
    ----------
    img_path:string
        需要溯源的图像URL地址
    k:int
        嵌入特征区域的个数，默认k=10
    th:
        水印水印提取阈值。默认：r=15
        
    Return
    ----------
    list:string
        溯源信息
    """
    return model.globalExtract(img_path, msg_shape, k, th)
