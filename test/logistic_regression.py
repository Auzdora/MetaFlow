"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: logistic regression.py
    Description: This demo shows how to handle logistic regression
        problem using MetaFlow.

    Created by Melrose-Lbt 2022-5-19
"""
import math

from loss_fn import CrossEntropy, LossMSE
from layers import Sigmoid, Linear
from core import Modules, Tensor
from data import DataLoader, Dataset
import numpy as np
from opt import SGD
from matplotlib import pyplot as plt

data_path1 = './logistic regression/ex2data1.txt'
data_path2 = './logistic regression/ex2data2.txt'


xq = np.arange(-5, 5, 0.01)
y = 1/(1+np.exp(-xq))
plt.plot(xq, y)
plt.show()



def data_loader(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(',')
            line[0] = float(line[0])
            line[1] = float(line[1])
            line[2] = float(line[2])
            dataset.append(line)

    return np.array(dataset)


class Myset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __getitem__(self, index):
        input_data = np.expand_dims(self.data[:, :-1], axis=-1)
        label = self.data[:, -1]
        return input_data[index], label[index]

    def __len__(self):
        return len(self.data)


class Model(Modules):
    def __init__(self):
        self.layer1 = Linear(in_features=2, out_features=1)
        self.sig = Sigmoid()
        super(Model, self).__init__()

    def forward(self, x):
        x = self.layer1(x)
        x = self.sig(x)

        return x

def prepro(input, set):
    return (input - np.mean(set)) / (np.max(set) - np.min(set))

def process_data(dataset):
    dataset[:,0] = prepro(dataset[:,0], dataset[:, 0])
    dataset[:, 1] = prepro(dataset[:, 1], dataset[:, 1])
    return dataset


def show_data(dataset):
    data0 = []
    data1 = []
    for i in dataset:
        if i[-1] == 0:
            data0.append(i)
        else:
            data1.append(i)
    data0 = np.array(data0)
    data1 = np.array(data1)
    plt.plot(data0[:,0], data0[:, 1], 'o', data1[:,0], data1[:,1],'x')
    plt.show()


dataset1 = process_data(data_loader(data_path1))
show_data(dataset1)
dataset2 = process_data(data_loader(data_path2))
train_data1 = Myset(dataset1)
train_data2 = Myset(dataset2)
dataloader1 = DataLoader(train_data1, 2, shuffle=True)
dataloader2 = DataLoader(train_data2, 2, shuffle=True)
model = Model()
optimizer = SGD(model, lr=0.01)
model.get_model_info()
for epoch in range(400):
    mean_loss = 0
    cnt = 0
    for data, label in dataloader1:
        output = model(Tensor(data))
        loss = CrossEntropy(label, output)
        loss.backward()
        # Update gradients
        optimizer.update()
        mean_loss += loss.value
        cnt += 1
        loss.clear()

    print("epoch{}: loss:{}".format(epoch, mean_loss / cnt))
xs = [[[-1], [-2]]]
print(model(Tensor(xs)))
print(model.layer1.weight, model.layer1.bias)
x = np.arange(-0.5, 0.5, 0.01)
y = -(model.layer1.weight.value[0][0][0]/model.layer1.weight.value[0][0][1]) * x - model.layer1.bias.value[0][0] / model.layer1.weight.value[0][0][1]
plt.plot(x, y)

data0 = []
data1 = []
for i in dataset1:
    if i[-1] == 0:
        data0.append(i)
    else:
        data1.append(i)
data0 = np.array(data0)
data1 = np.array(data1)
plt.plot(data0[:, 0], data0[:, 1], 'o', data1[:, 0], data1[:, 1], 'x')
plt.show()


def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(Sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - Sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg
