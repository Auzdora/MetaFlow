"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: MNIST_model.py
    Description: This file defines MNIST neural network.

    Created by Melrose-Lbt 2022-4-5
"""
from core import Modules, Tensor
from layers import Linear, Sigmoid, Conv2D
from loss_fn import LossMSE
import numpy as np


class MNIST_Net(Modules):
    def __init__(self):
        self.linear1 = Linear(in_features=784, out_features=256)
        self.linear2 = Linear(in_features=256, out_features=128)
        self.linear3 = Linear(in_features=128, out_features=10)

        self.sigmoid = Sigmoid()
        super(MNIST_Net, self).__init__()

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.linear3(x)

        return x
    
    
class Conv_MINIST_Net(Modules):
    def __init__(self):

        super(Conv_MINIST_Net, self).__init__()


if __name__ == "__main__":

    x = Tensor(np.random.normal(loc=0., scale=1., size=(1, 784, 1)))
    net = MNIST_Net()
    y = net(x)
    label = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(y)
    loss = LossMSE(label, y)
    print(loss)