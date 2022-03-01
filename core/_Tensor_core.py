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


class Tensor:
    def __init__(self, *args, grad_require=False):
        self.parents = []
        self.children = []
        self.grad = None
        self.grad_require = grad_require
        self.grad_fn = None

        if len(args) > 1:
            self.value = self.compute_value(*args)
        elif len(args) == 1:
            # If data is one dimensional array, it will be converted to two dimension
            self.value = np.array(args[0], dtype=float)
            if len(self.value.shape) == 1:
                self.value = np.expand_dims(self.value, axis=1)
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
        """
            Core method. Compute forward to get Tensors' value from
        current Tensor.
        """
        pass

    def backward(self):
        """
            Core method. Compute bacward to get gradient(Jacobi) from
        current Tensor.
        """
        if len(self.children) == 0:
            if self.grad_fn == "add":
                self.grad = np.array(np.eye(self.shape[0]))
            if self.grad_fn == "matrix multiply":
                self.grad = 1
        for parent in self.parents:
            if parent.grad_require:
                parent.grad = self.grad * self.compute_jacobi(parent)
            else:
                continue
        pass

    def connect_tensor(self, *args):
        """
            Connect parent nodes and children nodes.
            :return: tensors value list
        """
        self.grad_require = True
        tensors = []
        for tensor in args:
            tensor.children.append(self)
            self.parents.append(tensor)
            tensors.append(tensor.value)
        return tensors

    # Tensor op
    def sum(self, inplace=False):
        """
            Add all of the values in a single Tensor.
            :param inplace: bool -> True or False
            If inplace is True, the function return a modified Tensor on the
        same memory address.
            If inplace is False, the function return a newly created Tensor
        on a different memory address.
        :return: a Tensor
        """
        if inplace:
            self.value = self.value.sum()
            return self
        else:
            return Tensor(self.value.sum())
    # TODO: Add average
    # TODO: Add max
    # TODO: Add min
    # TODO: Add argmax
    # TODO: ...

    @abc.abstractmethod
    def compute_value(self, *args):
        """
            Compute value of a tensor from its parent nodes. And return
        it to self.value. Could be overwritten when necessary.
        :return:
        """

    @abc.abstractmethod
    def compute_jacobi(self, parent):
        """
            Compute gradient based on its children tensors. And return
        it to self.grad. Could be overwritten when necessary.
        :return:
        """


if __name__ == "__main__":
    a = np.array([1,2,3])
    a = Tensor(a)
    print(a)
