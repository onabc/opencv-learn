import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 轮廓正常展示
img = cv.imread('../assets/contours2.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)  # 大于17的取255，小于127的取0
binary, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
draw_img1 = img.copy()  # 若不用拷贝后的，而是用原图画轮廓，则画轮廓图绘把原始的输入图像重写，覆盖掉
res1 = cv.drawContours(draw_img1, contours, -1, (0, 0, 255), 2)

plt.subplot(2, 2, 1), plt.imshow(res1, cmap='gray')
plt.title('正常'), plt.xticks([]), plt.yticks([])

cnt = contours[0]

# 轮廓近似展示
epsilon = 0.05 * cv.arcLength(cnt, True)  # 周长的百分比，这里用 0.1 的周长作阈值
approx = cv.approxPolyDP(cnt, epsilon, True)  # 第二个参数为阈值
draw_img2 = img.copy()
res2 = cv.drawContours(draw_img2, [approx], -1, (0, 0, 255), 2)
plt.subplot(2, 2, 2), plt.imshow(res2, cmap='gray')
plt.title('近似展示'), plt.xticks([]), plt.yticks([])

# 外接矩形
draw_img3 = img.copy()
x, y, w, h = cv.boundingRect(cnt)  # 可以得到矩形四个坐标点的相关信息
res3 = cv.rectangle(draw_img3, (x, y), (x + w, y + h), (0, 255), 2)
plt.subplot(2, 2, 3), plt.imshow(res3, cmap='gray')
plt.title('外接矩形'), plt.xticks([]), plt.yticks([])


hull = cv.convexHull(cnt)
draw_img4 = img.copy()
res4 = cv.drawContours(draw_img4, [hull], -1, (0, 0, 255), 2)
plt.subplot(2, 2, 4), plt.imshow(res4, cmap='gray')
plt.title('凸包算法'), plt.xticks([]), plt.yticks([])

plt.show()
