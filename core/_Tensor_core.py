"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _Tensor_core.py
    Description: This file defines basic and core data structure in MetaFlow
        —— Tensor. Tensor is base class for everything in MetaFlow, for inst
        -ance, neural network layers, data loader, operations and so on.
            Tensor class also provides some class method for simple tensor op
        -eration like add, multiply, matrix multiply and so on.

    Created by Melrose-Lbt 2022-2-28
"""
from typing import overload

import numpy as np
import abc
from numpy import ndarray


# TODO: Write Doc - Define coding style for class object (init, instantiate method define, abstractmethod define,
#  class method define)
class Tensor:
    def __init__(self, *args, grad_require=False):
        tensors = []
        self.parents = []
        self.children = []
        self.grad = None
        self.grad_require = grad_require
        self.grad_fn = None
        for arg in args:
            tensors.append(arg)
        if len(tensors) > 1:
            self.value = self.compute_value(*args)
        elif len(tensors) == 1:
            self.value = np.array(tensors[0], dtype=float)
        self.shape = self.value.shape

    def __str__(self):
        return "<{}, shape={}, dtype=Tensor.float>".format(self.value, self.shape)

    def get_parents(self):
        """
            Get this Tensor's parents list.
        """
        return self.parents

    def get_child(self):
        """
            Get this Tensor's children list.
        """
        return self.children

    def forward(self):
        pass

    def backward(self):
        pass

    # Tensor op
    def sum(self, inplace=False):
        """
            Add all of the values in a single Tensor.
            :param inplace: bool -> True or False
            If inplace is True, the function return a modified Tensor on the
        same memory address.
            If inplace is False, the function return a newly created Tensor
        on a different memory address.
        :return:
        """
        if inplace:
            self.value = self.value.sum()
            return self
        else:
            return Tensor(self.value.sum())

    @abc.abstractmethod
    def compute_value(self, *args):
        """
            Compute value of a tensor from its parent nodes. And return
        it to self.value. Could be overwritten when necessary.
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_jacobi(self):
        """
            Compute gradient based on its children tensors. And return
        it to self.grad. Could be overwritten when necessary.
        :return:
        """
        raise NotImplementedError

    # Set relationships between parents and children.
    def relationship(self, *args):
        self.grad_require = True
        for tensor in args:
            tensor.children.append(self)
            self.parents.append(tensor)


class my(Tensor):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    a = Tensor([1, 2, 3])
    print(id(a))
    b = Tensor([2, 2, 2])
    c = a.sum(inplace=True)
    print(id(a), id(c))
