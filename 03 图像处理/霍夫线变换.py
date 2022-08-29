import cv2 as cv
import numpy as np

img = cv.imread("../assets/dave.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 20, 120, apertureSize=3)

"""
在应用霍夫变换之前，请应用阈值或使用Canny边缘检测。
第二和第三参数分别是[Math Processing Error]和[Math Processing Error]精度。
第四个参数是阈值，这意味着应该将其视为行的最低投票。请记住，票数取决于线上的点数。因此，它表示应检测到的最小线长。
"""
img1 = img.copy()
lines = cv.HoughLines(edges, 1, np.pi / 180, 120)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv.imshow("img", img1)
cv.waitKey()
cv.destroyAllWindows()

"""
minLineLength - 最小行长。小于此长度的线段将被拒绝。 
maxLineGap - 线段之间允许将它们视为一条线的最大间隙
"""
img2 = img.copy()
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv.imshow("img", img2)
cv.waitKey()
cv.destroyAllWindows()
