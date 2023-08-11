#形态学处理
import cv2

#进行腐蚀处理
def erode(img,structure='ellipse',size=(5,5),iterations=1,gray=True):
    """
    :param img: 输入灰度图像图像
    :param structure: 结构元素的形状
    :param size: 结构元素的size
    :param iterations: 重复操作的次数
    :param gray: 读取的是否是灰度图像（最终都要转化为灰度图像）
    """
    if len(img.shape) != 2:  # 确保是二维的，如果原来输入就是二维灰度图像即不需要操作
        img=img[:,:,0] if gray==True else cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #确保灰度图像
    if structure=='ellipse': #椭圆
        kerner=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
    elif structure=='rect': #矩形
        kerner = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    elif structure=='cross': #十字形
        kerner = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    erosion=cv2.erode(img,kernel=kerner,iterations=iterations)
    return erosion

#进行膨胀处理
def dilate(img,structure='ellipse',size=(5,5),iterations=1,gray=True):
    """
    :param img: 输入灰度图像图像
    :param structure: 结构元素的形状
    :param size: 结构元素的size
    :param iterations: 重复操作的次数
    :gray: 读取的是否是灰度图像（最终都要转化为灰度图像）
    """
    if len(img.shape) != 2:  # 确保是二维的，如果原来输入就是二维灰度图像即不需要操作
        img=img[:,:,0] if gray==True else cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #确保灰度图像
    if structure=='ellipse': #椭圆
        kerner=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
    elif structure=='rect': #矩形
        kerner = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    elif structure=='cross': #十字形
        kerner = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    dilate_img=cv2.dilate(img,kernel=kerner,iterations=iterations)
    return dilate_img

#进行开运算处理（先进行腐蚀操作再进行膨胀操作）
def opening(img,structure='ellipse',size=(7,7),gray=True):
    """
    :param img: 输入灰度图像图像
    :param structure: 结构元素的形状
    :param size: 结构元素的size
    :gray: 读取的是否是灰度图像（最终都要转化为灰度图像）
    """
    if len(img.shape) != 2:  # 确保是二维的，如果原来输入就是二维灰度图像即不需要操作
        img=img[:,:,0] if gray==True else cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #确保灰度图像
    if structure=='ellipse': #椭圆
        kerner=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
    elif structure=='rect': #矩形
        kerner = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    elif structure=='cross': #十字形
        kerner= cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    opening_img=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel=kerner)
    return opening_img

#进行闭运算处理（先进行膨胀操作再进行腐蚀操作）
def closing(img,structure='ellipse',size=(5,5),gray=True):
    """
    :param img: 输入灰度图像图像
    :param structure: 结构元素的形状
    :param size: 结构元素的size
    :gray: 读取的是否是灰度图像（最终都要转化为灰度图像）
    """
    if len(img.shape)!=2: #确保是二维的，如果原来输入就是二维灰度图像即不需要操作
        img=img[:,:,0] if gray==True else cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #确保灰度图像
    if structure=='ellipse': #椭圆
        kerner=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
    elif structure=='rect': #矩形
        kerner = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    elif structure=='cross': #十字形
        kerner= cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    closing_img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel=kerner)
    return closing_img