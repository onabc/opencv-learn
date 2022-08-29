import numpy as np
import cv2 as cv

flags = [i for i in dir(cv) if i.startswith('COLOR_')]
print(flags)

cap = cv.VideoCapture("assets/opencv.jpg")

# 读取帧
_, frame = cap.read()
# 转换颜色空间 BGR 到 HSV
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

# 定义HSV中蓝色的范围
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
# 设置HSV的阈值使得只取蓝色
mask = cv.inRange(hsv, lower_blue, upper_blue)
# 将掩膜和图像逐像素相加
res = cv.bitwise_and(frame, frame, mask=mask)

cc = np.hstack((frame, res))
cv.imshow('res', cc)
cv.waitKey()
cv.destroyAllWindows()
