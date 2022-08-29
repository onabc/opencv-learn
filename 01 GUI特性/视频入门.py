import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("camera can not open")
    exit()
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame, 1)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# 完成所有操作后，释放捕获器
cap.release()
cv.destroyAllWindows()

"""
cap = cv.VideoCapture("assets/数码宝贝.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
"""

"""
cap = cv.VideoCapture("assets/数码宝贝.mp4")
# 定义编解码器并创建VideoWriter对象
fourcc = cv.VideoWriter.fourcc(*'MP4V')
out = cv.VideoWriter('./assets/output.mp4', 0x7634706d, 20.0, (640,  480))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    out.write(frame)
# 完成工作后释放所有内容
cap.release()
out.release()
cv.destroyAllWindows()
"""