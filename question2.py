import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 存储每张图片的过滤后的色块数目
blobCounts = []

# 存储每个图像中每个色块的中心点坐标


centerPoints_list = []
# 遍历200张图片
for i in range(1, 201):
    centerPoints = []
    # 读取图像
    src = cv2.imread(f"C:/Users/Lenovo/Desktop/Attachment/Attachment/Attachment 1/{i}.jpg")

    # 高斯滤波
    blurredImage = cv2.GaussianBlur(src, (3, 3), 0)

    # R通道减去G通道
    diffImage = cv2.subtract(src[:, :, 2], src[:, :, 1])

    # 二值化
    thresholdValue = 64  # 阈值
    maxValue = 255  # 最大值
    _, binaryImage = cv2.threshold(diffImage, thresholdValue, maxValue, cv2.THRESH_BINARY)

    # 连通组件分析
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImage)

    # 过滤小色块并统计数量
    minAreaThreshold = 10  # 最小面积阈值
    filteredBlobCount = 0
    filteredImage = cv2.bitwise_not(np.zeros(src.shape[:2], dtype=np.uint8))

    for j in range(1, numLabels):
        area = stats[j, cv2.CC_STAT_AREA]

        if area >= minAreaThreshold:
            mask = (labels == j).astype(np.uint8)
            filteredImage = cv2.bitwise_and(filteredImage, mask * 255)
            filteredBlobCount += 1

            # 计算中心点坐标
            center_x = int(centroids[j, 0])
            center_y = 185 - int(centroids[j, 1])
            centerPoints.append((center_x, center_y))

    blobCounts.append(filteredBlobCount)
    centerPoints_list.append(centerPoints)
pd.DataFrame(centerPoints_list).to_csv('centerPoints.csv', index=False, header=False)