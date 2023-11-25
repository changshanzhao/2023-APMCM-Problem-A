import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calculate_apple_volume(radius, depth):
    # 根据近似的圆半径和相对深度计算苹果的体积
    volume = 0.0002153 * radius**3 + 0.02486 * radius**2
    volume = volume / depth

    return volume
# 遍历200张图片
list_m_tatol = []
for i in range(1, 201):
    list_m = []
    # 读取图像
    src = cv2.imread(f"C:/Users/Lenovo/Desktop/Attachment/Attachment/Attachment 1/{i}.jpg")
    # 读取深度图像
    depth = np.load(f"C:/Users/Lenovo/Desktop/Attachment/Attachment/Attachment 1/{i}_disp.npy")
    # 高斯滤波
    blurredImage = cv2.GaussianBlur(src, (3, 3), 0)

    # R通道减去G通道
    diffImage = cv2.subtract(src[:, :, 2], src[:, :, 1])

    # 二值化
    thresholdValue = 64  # 阈值
    maxValue = 255  # 最大值
    _, binaryImage = cv2.threshold(diffImage, thresholdValue, maxValue, cv2.THRESH_BINARY)
    edges = cv2.Canny(binaryImage, 5, 15)
    # 霍夫拟合圆
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=15, param2=20, minRadius=0,
                               maxRadius=0)
    # 提取圆的信息
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[1]-1, circle[0]-1)
            radius = circle[2]

            # V = 0.0002153*R^3 + 0.02486*R^2 引用自基于机器视觉的苹果重量检测研究
            V = calculate_apple_volume(radius*4.5, depth[center])
            # 苹果密度是0.9
            m = V * 0.9
            if m >= 80 and m <= 220:
                list_m.append(m)
    list_m_tatol.append(list_m)
pd.DataFrame(list_m_tatol).to_csv('m.csv', index=False, header=False)






