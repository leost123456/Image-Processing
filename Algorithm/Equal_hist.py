import cv2
import numpy as np
import os

#进行全局直方图均衡(HE)
def Equal_Hist(img):
    # 进行直方图均衡化
    b, g, r = cv2.split(img)  # 将图像分解出三个通道
    b_equal = cv2.equalizeHist(b)
    g_equal = cv2.equalizeHist(g)
    r_equal = cv2.equalizeHist(r)
    equal_img = cv2.merge([b_equal, g_equal, r_equal])  # 进行合成
    return equal_img

#进行对比度限制自适应直方图均衡（CLAHE）
def CLAHE(img,cliLimit=1.0,tileGridSize=(7, 7),**kwargs):
    """
    :param img: 读取的图像
    :param cliLimit: 控制对比度的参数，越大，对比度越强，同时也会引入噪声
    :param tileGridSize: 图像分割的局部块的大小
    :return:
    """
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=cliLimit, tileGridSize=tileGridSize)

    #进行均衡化
    b, g, r = cv2.split(img)  # 将图像分解出三个通道
    b_equal = clahe.apply(b)
    g_equal = clahe.apply(g)
    r_equal = clahe.apply(r)
    equal_img = cv2.merge([b_equal, g_equal, r_equal])  # 进行合成

    return equal_img

