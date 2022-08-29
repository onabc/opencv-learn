import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("assets/cat.jpg", cv.IMREAD_GRAYSCALE)

# 大于阈值的部分被置为255，小于部分被置为0
ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# 大于阈值部分被置为0，小于部分被置为255
ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
# 大于阈值部分被置为threshold，小于部分保持原样
ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
# 小于阈值部分被置为0，大于部分保持不变
ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
# 大于阈值部分被置为0，小于部分保持不变
ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(len(images)):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
