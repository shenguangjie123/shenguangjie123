import torch

# Axes3D可以画三维图
# 要保证数据密度，维度feature越多所需样本量越大（维度灾难）所以需要考虑降维
# back propagation反向传播，链式法则：路线上的偏导之乘积为总偏导
# Tensor Flow 用static graph，PyTorch用dynamic graph动态图，不过现在都是动态图了
# 训练集开发集测试集
# visdom可视化库  模型存盘防止运行到一半崩溃了
# 局部最优 优化算法
# 梯度下降算法 w=w-lr*grad(cost,w) lr为学习率要小，grad为梯度即导数，cost为总损失均值，w为所求参数
# 梯度为0的点为鞍点
# 学习率过大，会导致w在极小值点附近震荡，学习率过小，收敛速度慢，学习率要根据梯度下降的次数来调整，一般先用较大的学习率，然后逐渐减小
# 随机梯度下降，每次只用一个样本（防止遇到鞍点），收敛速度快，但是容易进入局部最优
# mini-batch梯度下降，每次用一部分样本，收敛速度比随机梯度下降慢，但是比梯度下降收敛速度快
# 激活函数：防止多层线性函数展开后导致神经网络退化为单层感知机，中间加激活函数以增加非线性，防止过拟合
# 计算图w->x->计算图（loss的函数）->loss，w为参数，x为初始输入，y为输出，loss对w求偏导，得到梯度，
x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # input
y_data = [9.0, 18.0, 31.0, 48.0, 69.0, 94.0]  # output

# tensor包含参数w与loss对w的比也即梯度
w1 = torch.tensor([3.0], requires_grad=True)  # tensor的requires_grad属性默认为False，不会计算loss关于w的梯度，直接用tensor会构建计算图
w2 = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
lr = 0.001   # lr不能太大，设置为0.01会反复震荡loss越来越大


# 定义模型（线性模型y=w1x^2+w2x+b） 2x^2+3x+4
def forward(x):
    return x ** 2 * w1 + x * w2 + b


# 定义损失函数：  MSE：n个（预测y-y）的平方和/n    这里是随机梯度下降算法
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2  # 这是在构建计算图而不是作运算


print("Predict (before training)", 7, forward(7).item())  # forward的结果是张量要取item变成标量

# 训练模型
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # 前馈只要计算loss
        l.backward()  # l为张量，backward为其成员函数，反向传播后会计算梯度，计算后计算图直接释放，下一次前馈重新计算计算图
        print("\tgrad: ", x, y, w1.grad.item(), w2.grad.item(), b.grad.item())  # item取出值变成python里的标量
        w1.data -= lr * w1.grad.data  # w是tensor，取data防止运算后构建计算图
        w2.data -= lr * w2.grad.data
        b.data -= lr * b.grad.data
        w1.grad.data.zero_()  # 梯度清零，防止梯度累加（每次新的l与w的梯度会加上前面算的梯度）
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print("progress:", epoch, l.item())

print("Predict (after training)", 7, forward(7).item())
