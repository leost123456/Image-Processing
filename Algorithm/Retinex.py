import cv2
import numpy as np

#SSR算法(注意尺度并没有还原)
def SingleScaleRetinex(img, sigma):
    """
    :param img: 输入图像
    :param sigma: 高斯标准差
    :return:
    """
    # 注意其中的cv2.GaussianBlur,如果标准差sigma很大的时候就是会添加高斯噪声，较小的时候可以去除高斯噪声，这里是要在原图片添加高斯噪声
    SSR = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return SSR

#MSR算法
def MultiScaleRetinex(img, sigma_list=[15, 80, 250]):
    """
    :param img: 输入图像
    :param sigma_list: 高斯标准差列表
    :return:
    """
    MSR= np.zeros_like(img)
    for sigma in sigma_list:
        MSR += SingleScaleRetinex(img, sigma)
    MSR = MSR / len(sigma_list)
    return MSR

#CR算法（主要用于MSRCR算法）
def ColorRestoration(img, alpha=125.0, beta=46.0):
    """
    :param img: 原始图像
    :param alpha: 经验参数1
    :param beta: 经验参数2
    :return:
    """
    img_sum = np.sum(img, axis=2, keepdims=True) #计算原始图片的各像素点BGR均值
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration

#SCB（简单色彩均衡，主要用于线性的放缩，按照一定的百分比去除最小和最大的部分）
def SimplestColorBalance(img, low_clip=0.01, high_clip=0.99):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True) #得到一个通道中各独立值和计数，从小到大进行排序
        current = 0 #目前的像素点个数（像素值大小从小到大排序）
        for u, c in zip(unique, counts): #每个独立像素值（从小到大）和对应的计数
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
        #将超过high_val的赋值high_val像素值,低于low_val的赋值low_val像素值(利用minimum和maximum十分巧妙)
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img

#MSRCR算法
def MSRCR(img, sigma_list=[15, 80, 250], G=5.0, b=25.0, alpha=125.0, beta=46.0, low_clip=0.01, high_clip=0.99):
    """
    :param img: 原始图像
    :param sigma_list: 高斯标准差列表
    :param G: 经验参数1
    :param b: 经验参数2
    :param alpha: CR的参数1
    :param beta: CR的参数2
    :param low_clip: 最小的百分比阈值（小于此百分比都会变为对应百分比的像素值例如0）
    :param high_clip: 最大的百分比阈值（大于此百分比都会变为对应百分比的像素值例如255）
    :return:
    """
    img = np.float64(img) + 1.0 #防止log=0

    MSR = MultiScaleRetinex(img, sigma_list) #得到MSR
    CR = ColorRestoration(img, alpha, beta) #得到CR系数
    MSRCR = G * (MSR * CR + b) #得到MSRCR

    #下面对每个通道进行归一化并线性扩展到0-255
    for i in range(MSRCR.shape[2]): #每个通道
        MSRCR[:, :, i] = (MSRCR[:, :, i] - np.min(MSRCR[:, :, i])) / \
                             (np.max(MSRCR[:, :, i]) - np.min(MSRCR[:, :, i])) * \
                             255

    MSRCR = np.uint8(np.minimum(np.maximum(MSRCR, 0), 255)) #确保像素值在0-255之间
    MSRCR = SimplestColorBalance(MSRCR, low_clip, high_clip) #抑制最小和最大的一部分数据（颜色均衡恢复）

    return MSRCR

#自适应MSRCR算法（不需要给其他参数）
def AutomatedMSRCR(img, sigma_list=[15, 80, 250]):
    img = np.float64(img) + 1.0

    img_retinex = MultiScaleRetinex(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break

        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)
    return img_retinex

#MSRCP算法
def MSRCP(img, sigma_list=[15,80,250], low_clip=0.01, high_clip=0.99):
    img = np.float64(img) + 1.0

    intensity = np.sum(img, axis=2) / img.shape[2]

    retinex = MultiScaleRetinex(intensity, sigma_list)

    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    intensity1 = SimplestColorBalance(retinex, low_clip, high_clip)

    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0

    img_msrcp = np.zeros_like(img)

    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    img_msrcp = np.uint8(img_msrcp - 1.0)

    return img_msrcp


