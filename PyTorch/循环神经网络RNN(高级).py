# -*- coding: utf-8 -*-
import csv
import gzip
import math
import time
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader


# 序列长短不一  输入名字，训练后输出国家
# 本质上是多维多分类问题，每个名字是一条记录，每个字母是一个维度，每个国家对应一类，名字字母的先后顺序也算是一种时间序列，所以用GRU（也可以RNN、LSTM）
# ASCII有128维 名字先转为字符列表再padding构造张量
# 国家再做一个索引标签
class NameDateset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'names_train.csv.gz' if is_train_set else 'names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:  # 不同类型的数据包的读取函数不同，pickle、hdfs、hd5等
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))
        self.country_dict = {c: i for i, c in enumerate(self.country_list)}  # 通过列表构造字典映射国家到索引
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        name = self.names[index]
        country = self.country_dict[self.countries[index]]
        return name, country

    def __len__(self):
        return len(self.names)

    def id2country(self, country_id):
        return self.country_list[country_id]

    def getCountriesNum(self):
        return self.country_num


BATCH_SIZE = 256

train_set = NameDateset(is_train_set=True)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_set = NameDateset(is_train_set=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

N_COUNTRY = train_set.getCountriesNum()  # 也就是输出的维度output_size
USE_GPU = False   # True 20个epoch用时5m27s False 20个epoch用时5m28s
# GPU设计用于高度并行的计算任务，这与深度学习中的矩阵乘法等操作非常契合，所以GPU通常比CPU在深度学习任务中更快
# 不过对于一些较小的模型或者数据集，使用CPU可能会更高效，因为模型和数据集小，GPU启动和数据传输的开销可能会超过其加速效果
# 如果GPU资源有限或者是较老的GPU型号，也可能无法发挥出其应有的加速效果
N_EPOCHS = 20   # 太大需要时间太久
N_CHARS = 128  # 也就是输入的维度input_size
HIDDEN_SIZE = 100
N_LAYER = 2  # 2层GRU
LR = 0.001


# Bi-direction意义是让未来的数据的影响也考虑进来，比如nlp里面后面的字也会影响当前字
# 所以双向的模型输出的部分有out和hidden两部分，hidden就是(hn_f, hn_b)，需要两个h0_f和h0_b，out就是两层的输出h合并(h1,h2,,,hn)
# h1=[h1_f,hn_b],h2=[h2_f,hn-1_b],,,hn=[hn_f,h1_b]


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layer, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_directions = 2 if bidirectional else 1
        self.output_size = output_size

        self.embedding = nn.Embedding(input_size, hidden_size)  # 相当于设置函数的系数而不是设置形参
        # input_size是输入:(inputSize,batchSize)，hidden_size是输出(inputSize,batchSize,hiddenSize)
        # 输入的seqLen是长短不一的，会先padding一下补0，再激活一下将所有数据转到0-1之间，这边的inputSize就是seqLen
        self.gru = nn.GRU(hidden_size, hidden_size, n_layer, bidirectional=bidirectional)
        # 输入inputs输出outputs维度都是hiddenSize:
        # inputs: input:(inputSize,batchSize,hiddenSize)
        #         hidden:(n_layer*n_directions,batchSize,hiddenSize)   h0
        # outputs: output:(inputSize,batchSize,hiddenSize*n_directions)   回归
        #          hidden:(n_layer*n_directions,batchSize,hiddenSize)     分类
        self.fc = nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layer * self.n_directions, batch_size, self.hidden_size)
        return hidden

    def forward(self, input, seq_lengths):  # seq_lengths是每条数据的长度的列表
        input = input.t()  # B x S -> S x B  B:batch,S:Seq(相对于inputSize)
        # print(input.shape)
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)  # 结果为(seqLength,batchSize,hiddenSize)

        gru_input = pack_padded_sequence(embedding, seq_lengths, batch_first=False, enforce_sorted=True)
        # 作用是去除embedding里padding的部分，根据seq_length即序列长度进行排序
        # 接下来每次取(batch_size,hidden_size)即batch_size个seq序列的第一项、第二项……拼接成为(seq_sum,hiddenSize)
        # 后面不足batch_size的部分也是直接拼接在后面保证seq_sum
        # 返回的是一个PackedSequence类的对象，不是tensor
        # output, hidden = self.gru(gru_input, hidden)   # 需要的是hidden，所以不用output
        _, hidden = self.gru(gru_input, hidden)
        # gru(RNN,LSTM也)可以接受PackedSequence对象，因为其本质上是(seq_length,batch_size,hidden_size)只不过经过了整理排序
        # print(hidden.shape)  !!!!!
        if self.n_directions == 2:
            hidden_cat = torch.cat((hidden[-1], hidden[-2]), dim=1)
            # hidden:(n_layer*n_directions,batchSize,hiddenSize) 将两个hidden按列拼接 !!!!!
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)  # ()!!!!!
        return fc_output


def name2list(name):
    arr = [ord(c) for c in name]
    return arr, len(arr)


def make_tensor(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [seq[0] for seq in sequences_and_lengths]  # 名字列表的列表
    seq_lengths = torch.LongTensor([seq[1] for seq in sequences_and_lengths])  # 名字列表的长度的列表
    countries = torch.LongTensor(countries)

    seq_tensor = torch.zeros(len(name_sequences), max(seq_lengths)).long()  # 先创建一个全0的张量
    for i, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[i, :seq_len] = torch.LongTensor(seq)  # 将已知的name列表贴到这个空张量上，从而创建了一个batchSize*max_len的张量
    # 排序，将最长的排在前面，返回排序好的和每个元素对应序号的列表
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)


def create_tensor(tensor):
    if USE_GPU:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tensor = tensor.to(device)
    return tensor


def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(train_loader, 0):
        inputs, seq_lengths, target = make_tensor(names, countries)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0 and i != 0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'{i * len(names)}/{len(train_set)}', end='')
            print(f' Loss: {total_loss / (i * len(inputs))}')

    return total_loss


def testModel():
    correct = 0
    total = len(test_set)
    print("evaluating trained model:", end='')
    with torch.no_grad():
        for i, (names, countries) in enumerate(test_loader, 1):
            inputs, seq_lengths, target = make_tensor(names, countries)
            output = classifier(inputs, seq_lengths)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

        percent = '{:.2f}'.format(100 * correct / total)
        print(f'Test set: Accuracy: {correct}/{total} {percent}%')

    return correct / total


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == '__main__':
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    if USE_GPU:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):   # 每一轮epoch都是用的同一个模型，只是模型参数不断优化以适应训练集，并且提升在测试集上的表现
        trainModel()
        acc = testModel()
        acc_list.append(acc)

    epoch = np.arange(1, N_EPOCHS + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.grid(True)  # 添加网格线
    plt.show()

# RNN的特点是处理序列与序列之间存在依赖关系，而CNN的特点是处理局部与局部之间存在依赖关系(卷积池化)，所以RNN适合处理序列数据，而CNN适合处理图像数据
# RNN处理的是序列数据(语音，语言等)，有记忆(使用相同rnnCell)和捕捉长期依赖(有前后顺序还能bidirectional)的能力，
# 并且可以处理不同长度的序列(pack_padded_sequence)，CNN只能处理固定大小的图像，
# RNN也会改变维度(embedding/fc)是为了解决分类问题(n_char->embedding->fc->n_class)
# embedding层是嵌入一个新维度，fc全连接层是改变一个维度的大小，二者都可以改变总维度
