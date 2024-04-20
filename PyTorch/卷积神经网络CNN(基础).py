# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# CNN适用于处理高维数据，如图像，音频等，图像就是C*W*H维数据，可以卷积为10维以便于分类
# 所以这个问题本质上是一个多维多分类问题
# 卷积convolution+下采样subsampling=特征提取器feature extraction
# 全连接fully connected=分类器classification
# RGB栅格图像（还有一种不会失真的图像叫矢量图像）：input channel（=3）*width*height  再对每个patch进行卷积运算（每一层即每一个channel与卷积核kernel数乘，再累加）
# 每个卷积核channel与输入一致，卷积核个数等于输出的channel：m*n*w*h(w*h为卷积核大小
'''
in_channels, out_channels = 1, 1
batch_size = 1
kernel_size = 3
wight, height = 5, 5

input = [3, 4, 6, 5, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]
# input = torch.randn(batch_size, in_channels, wight, height)   # 输入的channel必须一定
input = torch.Tensor(input).view(1, 1, 5, 5)   # B, C, W, H
conv_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, padding=1, bias=False, stride=1)
                            # 不需要偏置量设置为false，padding为1在输入input外面加一层0，使得输出还是5*5，步长stride指每次移动的步长
kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
conv_layer.weight.data = kernel.data  # weight是权重

output = conv_layer(input)

print(output)
# 下采样：最大池化层（常用）
maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)   # 步长默认与kernel_size一致
output = maxpool(output)
print(output)
'''

batch_size = 32
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])   # 将图像张量转换为0-1
# 先totensor为张量，再归一化为0-1分布即normalise，mean和std是均值和标准差
train_dataset = datasets.MNIST(root='../dataset/mnist', train=True, transform=transform, download=True)    # 0-256灰度级转为0-1或-1-1的张量
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)   # 数据集太大所以dataloader
test_dataset = datasets.MNIST(root='../dataset/mnist', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)     # b*1*28*28, 10个5*5的卷积核 ->b*10*24*24 池化一次->b*10*12*12
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)    # b*10*12*12, 20个5*5的卷积核 ->b*20*8*8 池化二次->b*20*4*4-> view一下为b*320
        # self.conv2_drop = torch.nn.Dropout2d()
        self.pooling = torch.nn.MaxPool2d(kernel_size=2, stride=2)    # 没有权重，一次就行，有权重则每次卷积后需要实例
        # self.fc = torch.nn.Linear(320, 10)  # b*320->b*10  实际情况可以先不算这块，输出一下结果的维度 要有增量式开发的思维
        self.l1 = torch.nn.Linear(320, 160)
        self.l2 = torch.nn.Linear(160, 80)
        self.l3 = torch.nn.Linear(80, 10)

    def forward(self, x):
        x = self.pooling(F.relu(self.conv1(x)))   # 卷积、非线性变换、池化  先池化再非线性变换计算量更小
        x = self.pooling(F.relu(self.conv2(x)))
        x = x.view(-1, 320)   # flatten展平
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)    # 因为要计算交叉熵损失，最后一层不需要激活，里面计算softmax的时候包含激活的步骤
        return x


model = Net()
decive = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    # 迁移运算到显卡上
model.to(decive)  # 模型放到显卡上

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # momentum 冲量


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(decive), target.to(decive)   # 模型和数据一定要放在同一块显卡上，这里把训练数据放在显卡上
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()   # loss是张量
        if batch_idx % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 100))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():   # 下面的都不用算梯度
        for data in test_loader:
            images, labels = data
            images, labels = images.to(decive), labels.to(decive)    # 测试数据放在显卡上
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)   # dim=0为行即第几个样本，dim=1为列即第几个特征，最大的既是预测的值/类
            total += labels.size(0)  # n*1的size，所以为n
            correct += (predicted == labels).sum().item()

    print('Accuracy on test set: %d %%' % (100 * correct / total))
    return correct/total


if __name__ == '__main__':
    epoch_list = []
    acc_list = []
    for epoch in range(10):
        train(epoch)
        acc = test()     # 准确率达到新高度可以先进行存盘
        epoch_list.append(epoch)
        acc_list.append(acc)

    plt.plot(epoch_list, acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

# 图像特征提取 自动特征提取cnn
# CNN将重点放在构建网络上，可以通过降低维度来解决分类问题，通常用于分类问题
# 但是CNN也可以通过一些方法例如将输出层修改为全连接层(其维度取决于需要)，损失函数使用MSE/MAE而非BCE/NLL，再将输出的连续数据标准化，
# 评价指标也使用MSE/RMSE/MAE等，通过这些方法用CNN来解决回归问题

# 卷积核的参数共享，减少了参数数量，避免过拟合
# 参数共享和最大池化层使得CNN具有平移不变性，图像平移不会影响卷积结果
# 多层的卷积池化可以捕捉图像局部特征，学习有代表性的特征


