"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _Optimizers.py
    Description: This file provides different types of optimizers.

    Created by Melrose-Lbt 2022-3-5
"""
import abc


class Optimizer(abc.ABC):
    """
        Abstract class for optimizers.
    """
    def __init__(self, learning_rate=0.01):
        # Access to all the leaf nodes that are gradable.
        # Number of gradable leaf nodes.
        self.leaf_num = None
        self.learning_rate = learning_rate

    @abc.abstractmethod
    def update(self):
        """
            Params update method, need to be implemented when you create a
        sub class. Gradient descent is baseline.
        """
        raise NotImplementedError("Optimizer update method undefined, you have to write it.")


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
        pass


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
