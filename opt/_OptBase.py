"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _OptBase.py
    Description: This file provides optimizers' base class. All sub-optimizers should be
        inherited from this class.

    Created by Melrose-Lbt 2022-4-5
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
