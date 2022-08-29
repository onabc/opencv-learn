import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../assets/cat.jpg', cv.IMREAD_GRAYSCALE)

lower_reso = cv.pyrDown(img)

higher_reso = cv.pyrUp(lower_reso)