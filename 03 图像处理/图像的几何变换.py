import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

""" 缩放
img = cv.imread('assets/cat.jpg')
res1 = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
cv.imshow('img', res1)

height, width = img.shape[:2]
print(height, width)
res2 = cv.resize(img, (2 * width, 2 * height), interpolation=cv.INTER_CUBIC)
"""

# 偏移为(100, 50)
img = cv.imread("assets/cat.jpg", cv.IMREAD_GRAYSCALE)
rows, cols = img.shape
print(rows, cols)

M = np.float32([[1, 0, 100], [0, 1, 50]])
dst1 = cv.warpAffine(img, M, (cols, rows))
cv.imshow('img', dst1)
cv.waitKey(0)
cv.destroyAllWindows()

# 相对于中心旋转90度, 没有任何缩放比例
# cols-1 和 rows-1 是坐标限制
M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
dst2 = cv.warpAffine(img, M, (cols, rows))
cv.imshow('img', dst2)
cv.waitKey(0)
cv.destroyAllWindows()

"""
# 仿射变换
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv.getAffineTransform(pts1, pts2)
dst = cv.warpAffine(img, M, (cols, rows))
plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()
"""

# 透视变换
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv.getPerspectiveTransform(pts1, pts2)
dst = cv.warpPerspective(img, M, (300, 300))
plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()
