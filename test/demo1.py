"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: demo1.py
    Description: This file provides a demo for MetaFlow.

    Created by Melrose-Lbt 2022-3-6
"""
import time
from core import Modules, Tensor, Add, MatMul
from loss_fn import LossMSE
import numpy as np

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


# Create a model
class Model(Modules):
    def __init__(self):
        super(Model, self).__init__()
        self.w1 = Tensor.random((3, 3), grad_require=True)
        self.b1 = Tensor((3, 1), grad_require=True)
        self.w2 = Tensor((2, 3), grad_require=True)
        self.b2 = Tensor((2, 1), grad_require=True)
        self.w3 = Tensor((1, 2), grad_require=True)
        self.b3 = Tensor((1, 1), grad_require=True)

    def forward(self, i):
        i = MatMul(self.w1, i)
        i = Add(i, self.b1)
        i = MatMul(self.w2, i)
        i = Add(i, self.b2)
        i = MatMul(self.w3, i)
        i = Add(self.b3, i)
        return i


past = time.time()
model = Model()
# Start to train
for epoch in range(100):
    mean_loss = 0
    for i in range(len(train_set)):
        x = Tensor(np.array(train_set[i, :-1]).T)
        labels = np.expand_dims(np.array(train_set[i, -1]), axis=0)
        output = model(x)
        loss = LossMSE(labels, output)
        loss.backward()
        # Update gradients
        model.w1.value = model.w1.value - 0.01 * model.w1.grad.reshape(3, 3)
        model.w2.value = model.w2.value - 0.01 * model.w2.grad.reshape(2, 3)
        model.w3.value = model.w3.value - 0.01 * model.w3.grad.reshape(1, 2)
        model.b1.value = model.b1.value - 0.01 * model.b1.grad.reshape(3, 1)
        model.b2.value = model.b2.value - 0.01 * model.b2.grad.reshape(2, 1)
        model.b3.value = model.b3.value - 0.01 * model.b3.grad.reshape(1, 1)
        mean_loss += loss.value
        loss.clear()
    print("epoch{}: loss:{}".format(epoch, mean_loss / len(train_set)))

now = time.time()
print("run time:{}s".format(now - past))
# Some examples to predict
# TODO: Add predict method for model, train model, test model.
print(model(Tensor([prepro(158, height), prepro(47, weight), prepro(22, bfrs)])))
print(model(Tensor([prepro(178, height), prepro(90, weight), prepro(15, bfrs)])))
