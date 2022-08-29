import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img1 = cv.imread('../assets/car.png', cv.IMREAD_GRAYSCALE)
plt.hist(img1.ravel(), 256, [0, 256])

img2 = cv.imread('../assets/car.png', cv.IMREAD_COLOR)
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv.calcHist([img2], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])

# 掩码的应用
mask = np.zeros(img1.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv.bitwise_and(img1, img1, mask=mask)
# 计算掩码区域和非掩码区域的直方图
# 检查作为掩码的第三个参数
hist_full = cv.calcHist([img1], [0], None, [256], [0, 256])
hist_mask = cv.calcHist([img1], [0], mask, [256], [0, 256])
plt.subplot(221), plt.imshow(img1, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 256])
plt.show()

plt.show()
