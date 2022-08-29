import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 为了获得更高的准确性，请使用二进制图像。因此，在找到轮廓之前，请应用阈值或canny边缘检测
im = cv.imread('../assets/contours.png')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 160, 255, cv.THRESH_BINARY)

# CHAIN_APPROX_NONE   存储所有边界点
# CHAIN_APPROX_SIMPLE 删除所有冗余点并压缩轮廓，从而节省内存
binary, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

img = imgray.copy()
draw_img = cv.drawContours(img, contours, -1, (56, 56, 56), 2)
cv.imshow('res', draw_img)
cv.waitKey()
cv.destroyAllWindows()

cnt = contours[0]  # 通过轮廓索引，拿到该索引对应的轮廓特征
print(cv.contourArea(cnt))  # 该轮廓的面积
print(cv.arcLength(cnt, True))  # 该轮廓的周长，True表示闭合的
