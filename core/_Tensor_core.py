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
from numpy import ndarray
from ._Constants import OP_LIST, MODULE_LIST


class Tensor:
    def __init__(self, *args, grad_fn=None, grad_require=False, special_op=False):
        # The number of Tensors version control, this version does not support multiple params(above 2) operations.
        assert len(args) < 3, "Input over 2 tensors is not supported in this version yet."
        self.parents = []
        self.children = []
        self.grad = None
        self.grad_require = grad_require
        self.grad_fn = grad_fn
        # special_op is a params that need to be enabled at operator __init__ process.
        self.special_op = special_op
        self.value = self.value_config(*args)
        self.shape = self.value.shape

    def __str__(self):
        return "Tensor({}, shape={}, dtype=Tensor.float)".format(self.value, self.shape)

    def __call__(self, input_data):
        pass

    def value_config(self, *args):
        """
            This function handles Tensor's value problem. Because users or developers could
        input any kind of data structure, there are some of examples down here:
            1. [USERS]:  When you create a Tensor which your input is clear: a = Tensor([1,2,3]).
            In this situation, your input is a single list.
            2. [USERS]:  When you create a Tensor which your input is its shape: a = Tensor((2,3)).
            In this situation, your input is a single tuple.
            3. [DEVELOPERS]: When you create a sub class like Operator or Optimizer, you need to
            inherit from Tensor. These sub classes may have multiple input Tensors, for example,
            MatMul(Tensor1, Tensor2) / Add(Tensor1, Tensor2)
        :param args:
        :return: value
        """
        if self.grad_fn is None:
            # That means user want to create a Tensor variable, *args' type could be list and tuple.
            if isinstance(args[0], list) or isinstance(args[0], ndarray):
                assert len(args) == 1, "If you want to create a Tensor, you could only input one list instead of " \
                                       "others. "
                value = np.array(args[0], dtype=float)

                assert len(value.shape) > 0, "You have to use numpy create at least one dimensional vector instead " \
                                             "of a value. "
                # If data is one dimensional array, it will be converted to two dimension
                if len(value.shape) == 1:
                    return np.expand_dims(value, axis=1)
                return value

            elif isinstance(args[0], tuple):
                return np.random.random(args[0])

            elif isinstance(args[0], int) or isinstance(args[0], float):
                return np.expand_dims(np.array(args[0]), axis=0)

        elif self.grad_fn in OP_LIST:
            if self.special_op:
                if self.grad_fn == '<TensorSum>':
                    return np.expand_dims(self.compute_value(*args), axis=0)
                if self.grad_fn == '<TensorSigmoid>' or self.grad_fn == '<LossMSE>' or self.grad_fn == '<TensorExp>' \
                        or self.grad_fn == '<TensorSoftmax>' or self.grad_fn == '<CrossEntropyLoss>':
                    return self.compute_value(*args)
            else:
                return self.compute_value(*args)

        # elif self.grad_fn in MODULE_LIST:
        #     return self.compute_value(*args)

    def clear(self):
        """
            In this version, you need to call clear() with your end node in compute graph,
        for example:
            for epoch in range(100):
                output = model(x)
                loss = LossMSE()
                loss.backward()
                ...
                loss.clear()
            You need to call this at every end of epoch to make sure memory won't blow up.
            This function aims at decoupling each Tensor in compute graph, by doing this,
        every object's reference count will go down to zero after each epoch, so python
        could call 'def __del__(self)' method automatically to delete release memory space.
        """
        # Leaf node decoupling
        if len(self.parents) == 0:
            self.parents = []
            self.children = []
            return

        # Other node decoupling
        if len(self.parents) > 0:
            for parent in self.parents:
                parent.children = []
                parent.clear()
            self.parents = []

    def get_parents(self):
        """
            Get this Tensor's parents list.
        """
        if len(self.parents) == 0:
            return None
        return self.parents

    def get_children(self):
        """
            Get this Tensor's children list.
        """
        if len(self.children) == 0:
            return None
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
            if not ((len(self.shape) == 1 and self.shape[0] == 1) or (len(self.shape) == 0)):
                raise ValueError("grad can be implicitly created only for scalar outputs")

            if self.grad_fn == '<TensorAdd>':
                self.grad = np.array(np.eye(self.shape[0] * self.shape[1]))

            if self.grad_fn == '<TensorSigmoid>' or self.grad_fn == '<TensorExp>':
                self.grad = np.array(np.eye(self.shape[0] * self.shape[1]))

            if self.grad_fn == '<TensorSum>' or self.grad_fn == '<LossMSE>' or self.grad_fn == '<TensorMatMul>' \
                    or self.grad_fn == '<CrossEntropyLoss>':
                self.grad = 1

        # If this node doesn't have parent node, that means it has no need to backpropagation
        if len(self.parents) > 0:
            for parent in self.parents:
                if parent.grad_require:

                    if self.grad_fn == '<TensorConv>' and parent.grad_fn == '<TensorConv>':
                        jacobi = self.compute_jacobi(parent)
                        if len(self.grad.shape) == 2:  # means it connects Linear and Conv
                            self.grad = np.expand_dims(self.grad, axis=0).repeat(self.in_channels,axis=0)
                        padding_grad = np.einsum('ijk, ikl->ijl', self.grad, jacobi)
                        # eliminate padding
                        grad = np.reshape(padding_grad,
                            (self.in_channels, self.output_h + 2*self.padding, self.output_w+2*self.padding)) \
                            [:, self.padding:self.output_h + self.padding, self.padding:self.output_w + self.padding]
                        parent.grad = np.reshape(grad, (self.in_channels, 1, -1))

                    elif self.grad_fn == '<TensorConv>' and parent.grad_fn is None:
                        self.grad = np.expand_dims(self.grad, axis=1).repeat(self.in_channels, axis=1)
                        jacobi = self.compute_jacobi(parent)
                        self_grad = np.expand_dims(jacobi, axis=0).repeat(self.out_channels, axis=0)
                        grad = np.einsum('ijkl, ijlm->ijkm', self.grad, self_grad)
                        parent.grad = grad

                    else:
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

    @classmethod
    def random(cls, tuple, grad_require=False):
        return Tensor(np.random.random(tuple), grad_require=grad_require)


if __name__ == "__main__":
    pass
