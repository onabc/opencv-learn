import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 2D卷积
# 保持这个内核在一个像素上，将所有低于这个内核的25个像素相加，取其平均值，然后用新的平均值替换中心像素。它将对图像中的所有像素继续此操作
img = cv.imread('assets/opencv.jpg')
kernel = np.ones((5, 5), np.float32) / 25
dst = cv.filter2D(img, -1, kernel)
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

# 图像模糊
# 通过将图像与低通滤波器内核进行卷积来实现图像模糊
blur = cv.blur(img, (5, 5))
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

# 高斯模糊
blur = cv.GaussianBlur(img, (5, 5), 0)
# 中位模糊
median = cv.medianBlur(img, 5)
# 双边滤波
blur = cv.bilateralFilter(img, 9, 75, 75)
