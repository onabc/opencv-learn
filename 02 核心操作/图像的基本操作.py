import numpy as np
import cv2 as cv

img1 = cv.imread('../assets/cat2.jpg')
img2 = cv.imread('../assets/opencv.png')

# 想把logo放在左上角，所以创建了ROI
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]
# 现在创建logo的掩码，并同时创建其相反掩码
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)

mask_inv = cv.bitwise_not(mask)

# 现在将ROI中logo的区域涂黑
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)

# 仅从logo图像中提取logo区域
img2_fg = cv.bitwise_and(img2, img2, mask=mask)

# 将logo放入ROI并修改主图像
dst = cv.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst
cv.imshow('res', img1)
cv.waitKey(0)
cv.destroyAllWindows()
