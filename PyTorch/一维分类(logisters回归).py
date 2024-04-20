import torch
import torchvision
import torch.nn.functional as F

# train_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=True,download=True)
# test_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=False,download=True)
train_set = torchvision.datasets.CIFAR10(root='../dataset/cifar10', train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='../dataset/cifar10', train=False, download=True)
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


# logistics回归做的是二分类任务而不是回归任务，分为某类和分类正确的概率，于是把计算结果映射到0到1内，属于饱和函数，有极限
# logistic回归 1/(1+e^(-wx+b))  最典型的sigmoid函数
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))  # 非线性变换，sigmoid在torch.nn.functional里面
        return y_pred


model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(size_average=False)  # 这个损失函数计算的不是距离的差异而是分布的差异
# loss=-y*log(y_pred)-(1-y)*log(1-y_pred)交叉熵，因为乘以了-1，所以越小越好  BCE函数 binary classification二分类
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器

for epoch in range(500):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Output weight and bias 权重和偏差
print('w: ', model.linear.weight.item())
print('b: ', model.linear.bias.item())

# Test Model
x_test = torch.Tensor([5.0])
y_test = model(x_test)
print('y_pred: ', y_test.item())
# 1数据集2选择模型3损失函数4优化器5训练数据6训练次数

# 可视化训练结果
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))  # 200行1列
y_t = model(x_t)   # 预测值
y = y_t.detach().numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()
