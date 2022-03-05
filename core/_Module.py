"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _Module.py
    Description: This file provide root class for models.

    Created by Melrose-Lbt 2022-3-5
"""
import abc


class Modules:
    def __call__(self, x):
        """
            Call forward function and return
        :param x: Tensor input
        :return: self.forward(x)
        """
        return self.forward(x)

    @abc.abstractmethod
    def forward(self, x):
        """
            Here you need to define your model's compute process.
        :param x: Tensor input
        :return: Tensor output
        """
