import cv2 as cv
import numpy as np

x = np.uint8([250])
y = np.uint8([26])

# 模运算
print(x + y)

# 饱和运算
print(cv.add(x, y))
