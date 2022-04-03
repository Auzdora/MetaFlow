"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: linear_layers.py
    Description: Contains linear layers for building module.

    Created by Melrose-Lbt 2022-3-6
"""
from core import Tensor, Modules, F


class Linear(Modules):
    """
        Linear layers (Fully connected layers) for neural network.
    """

    def __init__(self, in_features, out_features, bias=True):
        self.core_module = True
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor((1, out_features, in_features), grad_require=True)
        if bias:
            # TODO: 1 could be expanded if coding for batch
            self.bias = Tensor((out_features, 1), grad_require=True)
        else:
            self.bias = None

        super(Linear, self).__init__(self.core_module)

    def forward(self, x):
        if isinstance(x, Tensor):
            return F.fully_connected_layer(x, self.weight, self.bias)
        else:
            x = Tensor(x)
            return F.fully_connected_layer(x, self.weight, self.bias)

    def _get_module_info(self):
        """
            Have to rewrite when you define a brand new layers in 'layers'
        directory.
            It could be used to show the model's parameters in defined way.
            It also has to return its parameters value for further function.
        """
        print('In features dim: {}       Out features dim: {}'.format(self.in_features, self.out_features))
        print("Weight:\n{}".format(self.weight.value))
        print("Bias:\n{}".format(self.bias.value))
