# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset   # 抽象类不能实例化
from torchvision import datasets
from torchvision import transforms

'''
train_dataset = datasets.MNIST(root='../dataset/mnist', train=True, transform=transforms.ToTensor(), download=True)   # 0-256灰度级转为0-1或-1-1的张量
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)   # 数据集太大所以dataloader
test_dataset = datasets.MNIST(root='../dataset/mnist', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)
'''


# epoch 所有样本均经过一次前馈反馈 10000
# batch-size 每次训练即一次前馈反馈所用的样本数量 1000
# iteration 训练轮数 10
# shuffle 打乱样本顺序 num_workers 读取batch内容是否多线程读取，几线程
# dataset dataloader

class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.7)


class DiabetesDataSet(Dataset):  # 数据集不大，可以直接读进来，太大可以用哪部分都哪部分防止内存开销过大
    def __init__(self, file_path):
        xy = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.len = xy.shape[0]

    def __getitem__(self, index):  # 根据索引返回数据的魔法方法
        return self.x_data[index], self.y_data[index]

    def __len__(self):   #  返回数据集大小的魔法方法
        return self.len


dataset = DiabetesDataSet('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

if __name__ == '__main__':  # 不封装多线程读取时可能会报错，因为win用spawn，linux用fork，c语言接口api不一样
    for epoch in range(100):
        for i, (inputs, labels) in enumerate(train_loader, 0):   # i是batch_id
            # inputs, labels = data   # 准备数据
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
