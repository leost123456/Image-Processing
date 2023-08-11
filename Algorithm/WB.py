import cv2
import numpy as np
from Equal_hist import Equal_Hist

#基于灰度世界假设的AWB算法，
def AWB_gray(img):
    """
    :param img: 已经读取后的图片
    :return:
    """
    equal_img=Equal_Hist(img) #进行直方图均衡

    #计算均值
    B_mean = np.mean(equal_img[:, :, 0])
    G_mean = np.mean(equal_img[:, :, 1])
    R_mean = np.mean(equal_img[:, :, 2])

    #灰色系数
    Gray_mean=(B_mean+G_mean+R_mean)/3

    #计算三个通道的增益系数
    KB=Gray_mean/R_mean
    KG=Gray_mean/G_mean
    KR=Gray_mean/R_mean

    #进行修正
    equal_img[:, :, 0] *= KB.astype(np.uint8)
    equal_img[:, :, 1] *= KG.astype(np.uint8)
    equal_img[:, :, 2] *= KR.astype(np.uint8)

    return equal_img

#基于完美反射算法的白平衡(可能需要改)
def AWB_perfect(img,radio=0.1):
    """
    :param img: 输入图像
    :param radio: 设定的前百分比
    :return:
    """
    # 将B、G、R、和sum放入一个元组中
    BGR_turtle=list(zip(img[:,:,0].ravel(),img[:,:,1].ravel(),img[:,:,2].ravel(),np.sum(img,axis=2).ravel()))
    BGR_sort=sorted(BGR_turtle,reverse=True,key=lambda x: x[3]) #根据RGB和进行降序
    # BGR和最大的值（作为最亮的点）
    MaxVal=BGR_sort[0] #各像素点最大的
    # 根据设定的百分阈值求出前百分之几的BGR和的均值,并根据最大的BGR得到增益系数
    mean_radio=int(radio*len(BGR_sort))
    gain_B = MaxVal[0] / np.mean([BGR_sort[i][0] for i in range(mean_radio)])
    gain_G = MaxVal[1] / np.mean([BGR_sort[i][1] for i in range(mean_radio)])
    gain_R = MaxVal[2] / np.mean([BGR_sort[i][2] for i in range(mean_radio)])
    #对原始图像进行修正
    img[:, :, 0] = np.clip(np.float64(img[:,:,0])*gain_B,0,255).astype(np.uint8)
    img[:, :, 1] = np.clip(np.uint8(np.float64(img[:,:,1])*gain_G)).astype(np.uint8)
    img[:, :, 2] = np.clip(np.uint8(np.float64(img[:,:,2])*gain_R)).astype(np.uint8)

    return img
