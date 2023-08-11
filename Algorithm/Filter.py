import cv2
import numpy as np

#均值滤波
def Mean_Blur(img,k_size=(5,5),*args,**kwargs):
    """
    :param img: 输入图像
    :param k_size: 滤波核的大小
    :return:
    """
    blurred=cv2.blur(img,k_size) # 使用5x5的内核进行均值滤波
    return blurred

#高斯滤波
def GaussianBlur(img,k_size=(5,5),std=0,*args,**kwargs):
    """
    :param img: 输入图像
    :param size: 核的大小
    :param std: 高斯核的标准差，如果为0表示自动计算
    :return:
    """
    blurred = cv2.GaussianBlur(img,k_size,std,*args,**kwargs)  # 使用5x5的内核进行高斯滤波,最后一个参数0表示自动计算一个合适的标准差值
    return blurred

#中值滤波
def MedianBlur(img,k_size=5):
    """
    :param img: 输入图像
    :param k_size: 卷积核的大小
    :return:
    """
    blurred=cv2.medianBlur(img,k_size)
    return blurred

#双边滤波（能够尽量保留图像中的边缘和纹理信息）
def BilateralFilter(img,k_size=9,sigma_color=75,sigma_space=75):
    """
    注释：其中，sigmaColor 和 sigmaSpace 是双边滤波器的两个关键参数，它们控制了滤波器的宽松程度和严格程度。
    sigmaColor 控制颜色空间的滤波程度，sigmaSpace 控制坐标空间的滤波程度。
    sigmaColor 和 sigmaSpace 越大，滤波器的宽松程度越大，图像的细节信息就越容易被保留。
    但是，如果 sigmaColor 和 sigmaSpace 过大，可能会导致图像模糊或失真。
    :param img: 输入图像
    :param k_size: 核的大小（一定要是奇数）
    :param sigma_color: 颜色空间滤波器的标准差。较大的值可以使滤波器更加宽松，较小的值可以使滤波器更加严格。
    :param sigma_space:坐标空间滤波器的标准差。较大的值可以使滤波器更加宽松，较小的值可以使滤波器更加严格。
    :return:
    """
    #参数9表示空间域核大小（卷积核大小），75表示灰度域标准差，用于控制滤波的强度。
    filtered=cv2.bilateralFilter(img,k_size,sigmaColor=sigma_color,sigmaSpace=sigma_space)
    return filtered

#拉普拉斯滤波器
def Laplace_Filter(img):
    """
    :param img: 读取的图像
    :return:
    """
    kernel=np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]])
    #进行卷积(注意也会自动进行padding操作)
    sharpened=cv2.filter2D(img,-1,kernel) #其中的第二个参数是深度ddepth， np.uint8、np.int16、np.float32 等数据类型，如果 ddepth=-1，则输出图像的深度与输入图像相同
    return sharpened

