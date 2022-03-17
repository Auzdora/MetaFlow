"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: activation_layers.py
    Description: Contains activation function layers for building module.
        For instance, sigmoid, ReLU, tanh and so on.

    Created by Melrose-Lbt 2022-3-17
"""
from core import Tensor, Modules, F


class Sigmoid(Modules):
    """
        Sigmoid layer, a core module,could be instantiated when users define
    their own model.
        Sigmoid function g(x) = 1 / ( 1 + e^-x).
        Its derivative function g'(x) = g(x) (1 - g(x)).
    """
    def __init__(self):
        self.core_module = True
        super(Sigmoid, self).__init__(self.core_module)

    def forward(self, x):
        if isinstance(x, Tensor):
            return F.sigmoid(x)
        else:
            x = Tensor(x)
            return F.sigmoid(x)

    def get_module_info(self):
        print("Sigmoid layer")


class ReLU(Modules):
    def __init__(self):
        self.core_module = True
        super(ReLU, self).__init__(self.core_module)

    def forward(self, x):
        pass

    def get_module_info(self):
        print("ReLU layer")
