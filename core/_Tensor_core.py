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

import numpy as np
import abc
from _Constants import OP_LIST


class Tensor:
    def __init__(self, *args, grad_require=False, special_op=False):
        # args insurance
        assert len(args) < 3, "Input over 2 tensors is not supported in this version yet."

        self.parents = []
        self.children = []
        self.grad = None
        self.grad_require = grad_require
        self.grad_fn = None

        if special_op is False:
            if len(args) > 1:
                self.value = self.compute_value(*args)
            elif len(args) == 1:
                # If data is one dimensional array, it will be converted to two dimension
                self.value = np.array(args[0], dtype=float)
                if len(self.value.shape) == 1:
                    self.value = np.expand_dims(self.value, axis=1)
        else:
            self.value = np.expand_dims(self.compute_value(*args), axis=0)
        self.shape = self.value.shape

    def __str__(self):
        return "Tensor({}, shape={}, dtype=Tensor.float)".format(self.value, self.shape)

    def value_config(self, *args):
        """
            This function handles Tensor's value problem. Because users or developers could
        input any kind of data structure, there are some of examples down here:
            1. USERS:  When you create a Tensor which your input is clear: a = Tensor([1,2,3]).
            In this situation, your input is a single list.
            2. USERS:  When you create a Tensor which your input is its shape: a = Tensor((2,3)).
            In this situation, your input is a single tuple.
            3. DEVELOPERS: When you create a sub class like Operator or Optimizer, you need to
            inherit from Tensor. These sub classes may have multiple input Tensors, for example,
            MatMul(Tensor1, Tensor2) / Add(Tensor1, Tensor2)
        :param args:
        :return:
        """
        if isinstance(args, list):
            pass
        elif isinstance(args, tuple):
            pass

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
        # If this graph node is last node in compute graph, that means it has no children
        if len(self.children) == 0:
            if self.grad_fn == '<TensorAdd>':
                self.grad = np.array(np.eye(self.shape[0]))

            if self.grad_fn == '<TensorSum>' or self.grad_fn == '<LossMSE>' or '<TensorMatMul>':
                self.grad = 1
        # If this node doesn't have parent node, that means it has no need to backpropagation
        if len(self.parents) > 0:
            for parent in self.parents:
                if parent.grad_require:
                    parent.grad = np.dot(self.grad, self.compute_jacobi(parent))
                    parent.backward()
                else:
                    continue
        else:
            return

    def connect_tensor(self, *args):
        """
            Connect parent nodes and children nodes. This function is specifically created
        for abstract method 'def compute_value(self, *args)' to call.
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
            If inplace is True, the function return a modified Tensor on the
        same memory address.
            If inplace is False, the function return a newly created Tensor
        on a different memory address.
        :param inplace: bool -> True or False
        :return: a Tensor
        """
        if inplace:
            self.value = np.expand_dims(np.array(self.value.sum()), axis=0)
            self.shape = self.value.shape
            return self
        else:
            return Tensor(np.expand_dims(np.array(self.value.sum()), axis=0))
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
        """

    @abc.abstractmethod
    def compute_jacobi(self, parent):
        """
            Compute gradient based on its children tensors. And return
        it to self.grad. Could be overwritten when necessary.
        """


if __name__ == "__main__":
    k = Tensor([2,2,3], [2,2,4], [2,3,4])
    print(k)
