import tensorflow as tf
import numpy as np
import time
import cv2 as cv
# 读取影像
path = 'E:/BARELAND/数据集/SZTAKI_AirChange_Benchmark/SZTAKI_AirChange_Benchmark/Archieve/'
img1 = cv.imread(path+'img1.bmp')
img2 = cv.imread(path+'img2.bmp')

img1 = cv.copyMakeBorder(img1, 2, 2, 2, 2, cv.BORDER_REFLECT)
img2 = cv.copyMakeBorder(img2, 2, 2, 2, 2, cv.BORDER_REFLECT)

img_row = img1.shape[0]   # 行
img_columns = img1.shape[1]     # 列

img1 = np.asarray(img1, np.float32)
img2 = np.asarray(img2, np.float32)


img_change = []
temp = np.longfloat(0)
for i in range(img_row):
    for j in range(img_columns):
            temp = np.square(img1[i][j][0]-img2[i][j][0]) + np.square(img1[i][j][1]-img2[i][j][1])\
                   + np.square(img1[i][j][2]-img2[i][j][2])
            temp = np.sqrt(temp)
            img_change.append(temp)

# img_change = np.asarray(img_change, np.float32)
# max_ = img_change.max()
# min_ = img_change.min()
max_ = max(img_change)
min_ = min(img_change)
print('max = ', max_, 'min = ', min_)
for i in range(len(img_change)):
    img_change[i] = (img_change[i]-min_)/(max_ - min_)*255

# 生成差异图和差异二值图
img_gray = [[0 for col in range(img_columns)] for row in range(img_row)]
img_gray = np.asarray(img_gray, np.float32)

k = 0
for i in range(img_row):
    for j in range(img_columns):
        img_gray[i][j] = img_change[k]
        k += 1

img_gray01 = cv.imread(path + 'FCM.bmp', 0)   # 使用一个比较粗的结果来筛选纯净样本
img_gray01 = cv.copyMakeBorder(img_gray01, 2, 2, 2, 2, cv.BORDER_REFLECT)
img_gray = np.asarray(img_gray, np.uint8)
img_gray01 = np.asarray(img_gray01, np.uint8)

# io.imsave(path + 'chayitu.bmp', img_gray)
# io.imsave(path + '2zhitu.bmp', img_gray01)
print("差异图，基础二值图生成完毕")

