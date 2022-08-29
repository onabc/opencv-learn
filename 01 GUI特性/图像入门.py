import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("assets/cat.jpg", cv2.IMREAD_GRAYSCALE)

"""
cv.imshow("img", img)
cv.imwrite("assets/cat2.jpg", img)
cv.waitKey()
cv.destroyAllWindows()
"""
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.show()
