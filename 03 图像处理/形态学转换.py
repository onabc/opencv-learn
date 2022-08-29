import cv2 as cv
import numpy as np

# 1. 腐蚀
"""
原始图像中的一个像素(无论是1还是0)只有当内核下的所有像素都是1时才被认为是1，否则它就会被侵蚀(变成0)。
结果是，根据内核的大小，边界附近的所有像素都会被丢弃。因此，前景物体的厚度或大小减小，或只是图像中的白色区域减小。
它有助于去除小的白色噪声(正如我们在颜色空间章节中看到的)，分离两个连接的对象等。
"""
img = cv.imread('../assets/j.png', cv.IMREAD_GRAYSCALE)
_, img = cv.threshold(img, 160, 255, 0)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
img_ret1 = cv.erode(img, kernel, iterations=1)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))  # ksize=7x7,
img_ret2 = cv.erode(img, kernel, iterations=1)
img_ret3 = cv.erode(img, kernel, iterations=2)  # ksize=7x7，腐蚀2次

# 2. 膨胀
dilation = cv.dilate(img, kernel, iterations=1)

res = np.hstack((img, img_ret1, img_ret2, img_ret3, dilation))
cv.imshow('res', res)
cv.waitKey(0)
cv.destroyAllWindows()

# 3. 开运算 (先腐蚀 后膨胀) 对于消除噪音很有用
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

# 4. 闭运算 (先膨胀 后腐蚀) 在关闭前景对象内部的小孔或对象上的小黑点时很有用
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

# 5. 形态学梯度
# 这是图像膨胀和腐蚀之间的区别, 结果将看起来像对象的轮廓
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

# 6. 顶帽 输入图像和图像开运算之差
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)

# 7. 黑帽 输入图像和图像闭运算之差
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)