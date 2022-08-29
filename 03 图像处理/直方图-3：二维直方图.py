import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../assets/wiki.png')

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
plt.imshow(hist, interpolation='nearest')
plt.show()
