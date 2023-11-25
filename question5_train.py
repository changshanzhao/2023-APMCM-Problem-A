# 导入必要库
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
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
# 读取数据集
transform = transforms.Compose([
    transforms.Resize((270, 180)),
    transforms.ToTensor()
])
root_dir = r'C:\Users\Simple\Desktop\Math Model\2023APMCM\Attachment\Attachment\Attachment 2'
dataset = CustomDataset(root_dir, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# 划分数据集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_set, test_set = random_split(dataset, [train_size, test_size])

# 加载数据集，设置batch_size
trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
testloader = DataLoader(test_set, batch_size=1, shuffle=True)

# 定义损失函数为交叉熵损失函数
Lossfun = nn.CrossEntropyLoss()

# 定义网络结构
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
# 利用GPU进行训练
net = CNN().to('cuda:0')
# 定义优化器及一些超参数
optimizer = optim.Adam(net.parameters(), lr=0.001)
# 迭代5轮
for epoch in range(5):
    # 初始化loss为0
    total_loss = 0.0
    for i,data in enumerate(trainloader):
        # 加载数据放到cuda里
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # 输入进模型，并计算loss
        pred = net(inputs)
        loss = Lossfun(pred, labels)
        # 梯度清空
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 优化器优化
        optimizer.step()
        # 计算total_loss
        total_loss += loss.item()
        # 2000轮打印输出一次
        if i % 2000 == 1999:
            print('[%d, %d] loss:%.3f' % (epoch+1, i+1, total_loss/2000))
            total_loss = 0.0

net.eval()
cnt = 0.0
# 用测试集测试并输出结果
for i,data in enumerate(testloader):
    inputs, labels = data
    # 数据放到cuda里
    inputs = inputs.cuda()
    labels = labels.cuda()
    # argmax取最高概率为结果
    outs = net(inputs).argmax()
    # 输出放回cpu比较一下，预测正确，cut+=1
    if outs.cpu() == labels.cpu():
        cnt += 1
# 输出准确率
print(cnt/len(testloader))
torch.save(net.state_dict(), 'model.pth')



