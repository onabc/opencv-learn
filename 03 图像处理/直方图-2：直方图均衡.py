import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../assets/wiki.png', cv.IMREAD_GRAYSCALE)

# 即使图像是一个较暗的图像(而不是我们使用的一个较亮的图像)，经过均衡后，我们将得到几乎相同的图像
# 直方图均衡
equ = cv.equalizeHist(img)
res = np.hstack((img, equ))  # stacking images side-by-side
cv.imshow("res", res)
cv.waitKey()
cv.destroyAllWindows()

# CLAHE（对比度受限的自适应直方图均衡）
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
res = np.hstack((img, cl1))
cv.imshow("res", res)
cv.waitKey()
cv.destroyAllWindows()
