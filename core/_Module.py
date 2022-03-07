"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _Module.py
    Description: This file provide root class for models.

    Created by Melrose-Lbt 2022-3-5
"""
import abc
from collections import OrderedDict
from ._Tensor_core import Tensor


class Modules:

    def __init__(self, core_module=False):
        # If core_module is True, that means this class is a layers module defined by developers
        # If core_module is False, that means this class is a user defined model
        self.core_module = core_module
        self._parameters = OrderedDict()
        """
            Accept key-word data structure from its child class.
        """

    def __call__(self, x):
        """
            Call forward function and return.
        :param x: Tensor input
        :return: self.forward(x)
        """
        self._parameters = OrderedDict()
        return self.forward(x)

    def __setattr__(self, key, value):
        """
            Record model's trainable parameters into self._parameters: -> dict
        """
        if isinstance(value, OrderedDict):
            if self.core_module:
                pass
            else:
                self.__dict__[key] = value
                # counter makes sure that self._parameters record different value
                module_layer_cnt = 0
                for layers in self.__dict__:
                    module_layer_cnt += 1

                    if isinstance(self.__dict__[layers], bool):
                        continue
                    else:
                        layers_dict = self.__dict__[layers].__dict__
                        for params_name in layers_dict:
                            if isinstance(layers_dict[params_name], Tensor):
                                if layers_dict[params_name].grad_require:
                                    self._parameters[params_name + str(module_layer_cnt)] = layers_dict[params_name]
                                else:
                                    continue
                            else:
                                continue
        else:
            self.__dict__[key] = value

    def get_names_tensors(self):
        for name in self.__dict__:
            yield name, self.__dict__[name]

    def get_parameters(self):
        for items in self._parameters:
            yield items, self._parameters[items]

    def parameters(self):
        """
            Get models trainable parameters.
        :return: Model params dict
        """
        for name, params in self.get_parameters():
            yield name, params

    @abc.abstractmethod
    def forward(self, x):
        """
            Here you need to define your model's compute process.
        :param x: Tensor input
        :return: Tensor output
        """
