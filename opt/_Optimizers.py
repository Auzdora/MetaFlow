"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _Optimizers.py
    Description: This file provides different types of optimizers.

    Created by Melrose-Lbt 2022-3-5
"""
import abc
from core import Modules


class Optimizer(abc.ABC):
    """
        Abstract class for optimizers.
    """
    def __init__(self, model, learning_rate=0.001):
        if learning_rate < 0.0:
            raise ValueError("learning rate value should be positive value.")
        assert isinstance(model, Modules),\
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


class BGD(Optimizer):
    """
        Batch Gradient Descent.
    """
    def update(self):
        pass


class SGD(Optimizer):
    """
        Stochastic Gradient Descent.
    """

    def update(self):
        for params in self.model.parameters():
            params[1].value = params[1].value - self.learning_rate * params[1].grad.reshape(params[1].shape)


class MiniBGD(Optimizer):
    """
        Mini-Batch Gradient Descent.
    """
    def __init__(self, batch_size, learning_rate):
        super(MiniBGD, self).__init__(learning_rate)
        self.batch_size = batch_size

    def update(self):
        pass


class Momentum(Optimizer):

    def update(self):
        pass


class AdaGrad(Optimizer):

    def update(self):
        pass


class RMSProp(Optimizer):

    def update(self):
        pass


class Adam(Optimizer):

    def update(self):
        pass
