import torch.nn as nn
import torch.nn.functional as F
class Vanillann(nn.Module):
    def __init__(self):
        super(Vanillann, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)#定义第一个卷积层
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)#定义第二个卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)#定义第一个全连接
        self.fc2 = nn.Linear(120, 84)#定义第二个全连接
        self.fc3 = nn.Linear(84, 10)#定义第三个全连接

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))#第一个卷积层激活并池化
        x = self.pool(F.relu(self.conv2(x)))#第二个卷积层激活并池化
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
