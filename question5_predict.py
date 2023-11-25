# 导入必要库
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.filepaths = []
        self.labels = []
        self.transform = transform

        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            file_list = os.listdir(class_dir)
            self.filepaths.extend([os.path.join(class_dir, file) for file in file_list])
            self.labels.extend([i] * len(file_list))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        file_path = self.filepaths[index]
        label = self.labels[index]
        image = Image.open(file_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 定义三种不同的卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        # 定义池化层
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 定义全连接层
        self.fc1 = nn.Linear(96480, 12800)
        self.fc2 = nn.Linear(12800, 1280)
        self.fc3 = nn.Linear(1280, 64)
        self.fc4 = nn.Linear(64, 5)

        # 定义激活函数
        self.relu = nn.ReLU()

    # 前向传播网络结构
    def forward(self, x):
        # 数组采样操作, 采样后的图片大小为32*32
        x = F.interpolate(x, size=(270, 180), mode='bilinear', align_corners=True)
        # 第一层卷积与池化
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        # 第二层卷积与池化
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        # 第三层卷积
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # 将张量展平
        # 利用全连接层将输出转化成所需的维度
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)

        return x
# 读取数据集
transform = transforms.Compose([
    transforms.Resize((270, 180)),
    transforms.ToTensor()
])
root_dir = r'C:\Users\Simple\Desktop\Math Model\2023APMCM\Attachment\Attachment\Attachment 3'
dataset = CustomDataset(root_dir, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
model = CNN().to('cuda:0')
model.load_state_dict(torch.load('model.pth'))
model.eval()
# predict题目给出数据
res_list = []
for i,data in enumerate(data_loader):
    inputs, labels = data
    # 数据放到cuda里
    inputs = inputs.cuda()
    # argmax取最高概率为结果
    outs = model(inputs).argmax()
    res_list.append(outs.cpu())
pd.DataFrame(res_list).to_csv("question5.csv", index=False, header=False)

