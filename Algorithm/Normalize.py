import cv2
import numpy as np

#minmax归一化并映射到指定的范围（针对图像）
def Min_Max_Normalize(img,low=0,high=255):
    """
    :param img: 输入图像
    :param low: 下限值
    :param high: 上限值
    :return:
    """
    #对全图的所有通道一起归一，要注意第二个参数一定要是None或者原图存储变量，不然会出问题
    new_img=cv2.normalize(img,None,low,high,cv2.NORM_MINMAX)
    return new_img

#切比雪夫归一化操作(注意没有缩放)
def Norm_Inf(img):
    new_img = cv2.normalize(img, None,cv2.NORM_INF)
    return new_img

#曼哈顿距离，L1范数归一化
def Norm_L1(img):
    new_img = cv2.normalize(img, None, cv2.NORM_L1)
    return new_img

#欧几里得距离，L2范数归一化
def Norm_L2(img):
    new_img = cv2.normalize(img, None, cv2.NORM_L2)
    return new_img


