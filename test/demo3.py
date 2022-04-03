"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: demo3.py
    Description: This demo tests how to use 'data' module to pack your data
        and feed them in to your neural networks.

    Created by Melrose-Lbt 2022-3-29
"""

import time

import numpy as np
from core import Modules, Tensor
from layers import Linear, Sigmoid
from loss_fn import LossMSE
from opt import SGD
from data import DataLoader, Dataset

# Prepare data
male_heights = np.random.normal(171, 2, 500)
female_heights = np.random.normal(158, 3, 500)

male_weights = np.random.normal(70, 5, 500)
female_weights = np.random.normal(57, 4, 500)

male_bfrs = np.random.normal(16, 2, 500)
female_bfrs = np.random.normal(22, 2, 500)

male_labels = [1] * 500
female_labels = [0] * 500

height = np.concatenate((male_heights, female_heights))
weight = np.concatenate((male_weights, female_weights))
bfrs = np.concatenate((male_bfrs, female_bfrs))
labels = np.concatenate((male_labels, female_labels))


def prepro(input, set):
    return (input - np.mean(set)) / (np.max(set) - np.min(set))


# Preprocess and shuffle
train_set = np.array([prepro(height, height), prepro(weight, weight), prepro(bfrs, bfrs), labels]).T


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
        self.layer1 = Linear(in_features=3, out_features=3)
        self.layer2 = Linear(in_features=3, out_features=2)
        self.layer3 = Linear(in_features=2, out_features=1)

        self.sig = Sigmoid()
        super(Model, self).__init__()

    def forward(self, x):
        x = self.layer1(x)
        x = self.sig(x)
        x = self.layer2(x)
        x = self.sig(x)
        x = self.layer3(x)

        return x


past = time.time()
model = Model()
train_data = Myset(train_set)
dataloader = DataLoader(train_data, 1, shuffle=True)
model.get_model_info()
optimizer = SGD(model, learning_rate=0.01)
# Start to train
for epoch in range(3000):
    mean_loss = 0
    cnt = 0
    for data, label in dataloader:
        output = model(Tensor(data))
        loss = LossMSE(label, output)
        loss.backward()
        # Update gradients
        optimizer.update()
        mean_loss += loss.value
        cnt += 1
        loss.clear()

    print("epoch{}: loss:{}".format(epoch, mean_loss / cnt))

now = time.time()
print("run time:{}s".format(now - past))
# Some examples to predict
# TODO: Add predict method for model, train model, test model.
print(model(Tensor([prepro(158, height), prepro(47, weight), prepro(22, bfrs)])))
print(model(Tensor([prepro(178, height), prepro(90, weight), prepro(15, bfrs)])))