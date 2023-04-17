import os
from osgeo import gdal
import numpy as np
import pandas as pd

def readTif(fileName):
#读取图像
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
        
    width = dataset.RasterXSize   #对应clo,x
    height = dataset.RasterYSize   #对应row, y
    bands = dataset.RasterCount     #波段数

    data = dataset.ReadAsArray()   #图像矩阵
    geotrans = dataset.GetGeoTransform()   #放射信息
    proj = dataset.GetProjection()    #投影
    return width, height, bands, data, geotrans, proj

def writeTiff(im_data, im_geotrans, im_proj, path):
#写入图像
    im_data[np.isinf(im_data)] = 0  #处理异常值
    im_data[np.isneginf(im_data)] = 0  #处理异常值
    im_data[np.isnan(im_data)] = 0   #处理异常值

    #判断图像像素位数
    if 'int8' in im_data.dtype.name:  
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
        
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

if __name__ == '__main__':
    fileName = r'E:\test\LT001_test.TIF'   #原始文件
    path = r'E:\test\LT001_res.TIF'    #生成的文件
    width, height, bands, data, geotrans, proj = readTif(fileName)
    print("width: " , width , " height: " , height , " bands:" , bands)
    print(type(data),data[...,0],len(data))
    #writeTiff(data,geotrans, proj, path)