import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

def map_to_custom_mean(data, target_mean):
    # 计算原始数据的均值和标准差
    original_mean = np.mean(data)
    original_std = np.std(data)

    # 进行Box-Cox变换
    transformed_data, _ = stats.boxcox(data)

    # 调整数学期望和标准差
    transformed_data = transformed_data * (original_std / np.std(transformed_data))
    transformed_data = transformed_data + (target_mean - np.mean(transformed_data))

    return transformed_data

def map_to_0_1(lst):
    min_value = min(lst)
    max_value = max(lst)
    mapped_lst = [(value - min_value) / (max_value - min_value) for value in lst]
    return mapped_lst
# 存储每张图片的过滤后的色块数目
means = []

# 遍历200张图片
for i in range(1, 201):
    centerPoints = []
    # 读取图像
    src = cv2.imread(f"C:/Users/Lenovo/Desktop/Attachment/Attachment/Attachment 1/{i}.jpg")

    # 高斯滤波
    blurredImage = cv2.GaussianBlur(src, (3, 3), 0)

    # R通道减去G通道
    diffImage = cv2.subtract(src[:, :, 2], src[:, :, 1])

    # 每个像素小于64则置为0
    # 然后将所有像素减去64，结果就是除了苹果的位置其他位置都是0
    diffImage = np.clip(diffImage, 64, None)
    diffImage = diffImage - 64

    # 取剩余像素平均值
    mean_value = np.mean(diffImage)
    means.append(mean_value)

# 将最小值变成1，不然不能进行Box-Cox变换
means = [i+1 for i in means]

# 将列表的值映射到分散的分布，并符合正态分布，数学期望设置为0.6，标准差与原始数据保持一致
mapped_list = map_to_custom_mean(means, 0.6)
# 线性变换到0-1之间
mapped_list = map_to_0_1(mapped_list)
pd.DataFrame(mapped_list).to_csv('means.csv', index=False, header=False)

