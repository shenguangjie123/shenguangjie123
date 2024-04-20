import torch

# 这里使用的是mini-batch梯度下降
x_data = torch.Tensor([[1.0], [2.0], [3.0]])  # 数据需要是矩阵
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
# 线性回归是线性模型，任务是回归

# 计算图Linear Unit  输入和输出的维度决定了计算图的形式
class LinearModel(torch.nn.Module):  # module里有很多方法，module构建了前馈后里面会自动构造反向传播
    def __init__(self):
        super(LinearModel, self).__init__()  # 父类的init  nn为neural network神经网络
        self.linear = torch.nn.Linear(1, 1)  # 1 input 维度 and 1 output 维度  y=Ax+b  线性模型
        # x的矩阵，行是样本列是feature，n*1为一维y=x*w或y=wT*w（转置的w），输入输出的x与y维度需要相同

    def forward(self, x):    # 覆盖父类里的forward
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)  # 求损失的函数，size_average=False 表示不求1/n
# =表示重写函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 找到model里的所有参数，有梯度的都加入训练
# 优化器还有Adagrad、Adam、Adamax、ASGD、LBFGS、RMSprop、Rprop等
# optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)

for epoch in range(500):
    y_pred = model(x_data)  # forward过程
    loss = criterion(y_pred, y_data)
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')   # 直接写loss会调用__str__()打印出来

    optimizer.zero_grad()   # 先对所有训练的参数进行梯度清零
    loss.backward()
    optimizer.step()   # 自动更新参数梯度

# Output weight and bias 权重和偏差   因为linear就是y=wx+b
print('w: ', model.linear.weight.item())
print('b: ', model.linear.bias.item())

# Test Model
x_test = torch.Tensor([5.0])
y_test = model(x_test)
print('y_pred: ', y_test.item())
