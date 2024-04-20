# -*- coding: utf-8 -*-
import torch

# dense/deep network 全连接网络 权重很多   卷积神经网络里面的权重是共享的
# rnn专门用于时间序列的数据（金融、天气、自然语言等） 也通过权重共享减少一些权重
# xt->RNN Cell(Linear)->ht(xt与ht维度可以不同) ht与下一个xt+1一起再输入到RNN Cell里面(线性变换和激活)得到ht+1
# 图像->CNN->FC(全连接层)->RNN->文章
# rnn里面用tanh函数激活更多（-1，1）   ht=tanh(w1*xt+b1+w2*ht-1+b2)

'''
h = torch.rand(size=(1, 10))
line = Leaner()
for x in X:
    h = line(x, h)  # 递归
'''
'''
batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

dataset = torch.randn(size=(seq_len, batch_size, input_size), requires_grad=True)
# numlayers 多少层rnn cell
hidden = torch.zeros(batch_size, hidden_size)

out = []
for idx, input in enumerate(dataset):
    print('='*20, idx+1, '='*20)
    print('Input size:', input.shape)

    hidden = cell(input, hidden)
    out.append(hidden)
    print('Output:', out)
    print('Hidden state:', hidden.shape)
    print('Hidden', hidden)
'''
# 文本转向量 独热向量 维度为字典大小
idx2char = ['e', 'h', 'l', 'o']
# 这里的文本生文本看似是多维回归问题，但是实际上是多维多分类问题，所以采用的是交叉熵CrossEntropyLoss计算损失
# 我们训练hello所属的类是ohloe即e类，但是我们实际用到的是ohloe这一字符串来计算交叉熵而不是结果的e，因为给的label是ohloe  ？？？输入前一个来预测下一个是什么类
'''
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 0]
one_hot_lookup = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = torch.tensor(x_one_hot).view(seq_len, batch_size, input_size)   # seq_len*batch_size*input_size
labels = torch.LongTensor(y_data)    # seq_len*batch_size   seq_len表示输入几个字母，input_size表示每个字母的独热码
'''
# 独热码的方式维度太高，如果是英文单词可能要几千几万
# 低维，稠密，学习导向的方法：embedding也就是数据降维(可升可降)，需要是长整型的张量，后面加rnn_cell和fc线性全连接层(使得输入输出一致)
x_data = [[1, 0, 2, 2, 3]]   # batch_Size,seq_len
y_data = [3, 1, 2, 3, 0]     # batch_size*seq_len
inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)

input_size = 4
embedding_size = 10
hidden_size = 8
num_class = 4    # labels类别数目
batch_size = 1
num_layers = 2   # 2层rnn
seq_len = 5
bidirectional = False


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.Rnn = torch.nn.RNN(input_size=embedding_size, hidden_size=hidden_size,num_layers=num_layers, batch_first=True)   # 输入为embedding的size
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, inputs):
        num_directions = 2 if bidirectional else 1
        hidden = torch.zeros(num_layers*num_directions, inputs.size(0), hidden_size)   # 因为batch_first=True，所以inputs的第一个维度是batch_size
        inputs = self.emb(inputs)
        # 输入的inputs是seq_len*batch_size，输出是seq_len*batch_size*embedding_size
        out, _ = self.Rnn(inputs, hidden)          # out是h的列表，_是最后的ht
        # h列表: 输出为seq_len*batch_size*hidden_size(embedding_size->hidden_size)
        # ht: 输出为num_layers*num_directions*batch_size*hidden_size(seq_len->num_layers*num_directions,embbeding_size->hidden_size)
        # 总之embedding一个维度再转化为hidden_size
        out = self.fc(out)    # 将RNN层输出的out转化为适合类别预测的形式，也就是将hidden中的hidden_size转为num_class
        return out.view(-1, num_class)   # 输出需要调整成适当的形式和labels计算交叉熵时可以相对应
        # inputSize->hiddenSize->numClass/labelNum


net = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)    # Adam优化器

for epoch in range(10):
    optimizer.zero_grad()
    outputs = net(inputs)
    print('predicted string: ', end='')
    # print(outputs.size(), labels.size())
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    print(''.join([idx2char[i.item()] for i in idx]), end='')
    print(', Epoch [{0:d}/10] loss: {1:.4f}'.format(epoch+1, loss.item()))

# LSTM长短时计算网络 效果比RNN好得多但是计算复杂性能低
# 遗忘门乘以一个0到1的数减少信息，不过解释没有很明确 还有一些卷积池化激活的操作 也可以减少一些梯度消失的影响
# GRU是RNN和LSTM的结合体，比RNN快，比LSTM简单
