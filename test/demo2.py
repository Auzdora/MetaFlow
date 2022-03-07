"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: demo2.py
    Description: This file provides a demo for MetaFlow.

    Created by Melrose-Lbt 2022-3-6
"""
import time

import numpy as np
from core import Modules, Tensor
from layers import Linear
from loss_fn import LossMSE
from opt import SGD

# Prepare data
male_heights = np.random.normal(171, 6, 500)
female_heights = np.random.normal(158, 5, 500)

male_weights = np.random.normal(70, 10, 500)
female_weights = np.random.normal(57, 8, 500)

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
np.random.shuffle(train_set)


class Model(Modules):
    def __init__(self):
        self.layer1 = Linear(in_features=3, out_features=3)
        self.layer2 = Linear(in_features=3, out_features=2)
        self.layer3 = Linear(in_features=2, out_features=1)
        super(Model, self).__init__()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


past = time.time()
model = Model()
model.parameters()
optimizer = SGD(model, learning_rate=0.01)
# Start to train
for epoch in range(100):
    mean_loss = 0
    for i in range(len(train_set)):
        x = Tensor(np.array(train_set[i, :-1]).T)
        labels = np.expand_dims(np.array(train_set[i, -1]), axis=0)
        output = model(x)
        time2 = time.time()
        loss = LossMSE(labels, output)
        loss.backward()
        # Update gradients
        optimizer.update()
        mean_loss += loss.value
        loss.clear()

    print("epoch{}: loss:{}".format(epoch, mean_loss / len(train_set)))

now = time.time()
print("run time:{}s".format(now - past))
# Some examples to predict
# TODO: Add predict method for model, train model, test model.
print(model(Tensor([prepro(158, height), prepro(47, weight), prepro(22, bfrs)])))
print(model(Tensor([prepro(178, height), prepro(90, weight), prepro(15, bfrs)])))
