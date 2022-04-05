"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _Optimizers.py
    Description: This file provides different types of optimizers.

    Created by Melrose-Lbt 2022-3-5
"""
import abc

import numpy as np

from core import Modules

beta = 0.9  # momentum decay ratio / RMSProp decay ratio
gama = 1e-10  # infinite min value


class Optimizer(abc.ABC):
    """
        Abstract class for optimizers.
    """

    def __init__(self, model, learning_rate=0.001):
        if learning_rate < 0.0:
            raise ValueError("learning rate value should be positive value.")
        assert isinstance(model, Modules), \
            "input model is not a Modules class."
        # Access to all the leaf nodes that are gradable.
        # Number of gradable leaf nodes.
        self.model = model
        self.learning_rate = learning_rate

    @abc.abstractmethod
    def update(self):
        """
            Params update method, need to be implemented when you create a
        sub class. Gradient descent is baseline.
        """
        raise NotImplementedError("optimizer update method undefined, you have to write it.")


class SGD(Optimizer):
    """
        Stochastic Gradient Descent.
    """

    def __init__(self, model, lr=0.001, ratio_decay=False, momentum=False, adagrad=False, rmsprop=False):
        # model and learning rate
        self.model = model
        self.lr = lr

        # different optimization method button
        self.ratio_decay = ratio_decay
        self.momentum = momentum
        self.AdaGrad = adagrad
        self.RMSProp = rmsprop

        self.v = dict()  # velocity of momentum
        self.s = dict()  # state of AdaGrad and RMSProp

        super(SGD, self).__init__(model, learning_rate=lr)

    def update(self):
        """
            Stochastic gradient descent optimizer update process. This method congregates naive
        SGD, Momentum, Adagrad and RMSProp method into a single function. At same time it prov-
        ides decay ratio for SGD.
        """
        for name, params in self.model.parameters():
            g = params.grad.reshape(params.shape)
            lg = self.lr * params.grad.reshape(params.shape)
            if self.AdaGrad or self.RMSProp:
                g_2 = np.power(g, 2)

            if self.momentum:
                if name not in self.v:
                    self.v[name] = g
                else:
                    self.v[name] = beta * self.v[name] - lg
                params.value = params.value + self.v[name]

            elif self.AdaGrad:
                if name not in self.s:
                    self.s[name] = g_2
                else:
                    self.s[name] = self.s[name] + g_2
                params.value = params.value - lg / np.sqrt(self.s[name] + gama)
            elif self.RMSProp:
                if name not in self.s:
                    self.s[name] = g_2
                else:
                    self.s[name] = beta * self.s[name] + (1 - beta) * g_2
                params.value = params.value - lg / np.sqrt(self.s[name] + gama)

            else:
                params.value = params.value - lg


class AdaGrad(Optimizer):

    def update(self):
        pass


class RMSProp(Optimizer):

    def update(self):
        pass


class Adam(Optimizer):

    def update(self):
        pass


if __name__ == "__main__":
    x = dict()

    name = ['1', '2', '3', '4', '5', '6', '7']
    v = 0

    for i in name:
        x[i] = v
        v += 2

    print(x)
