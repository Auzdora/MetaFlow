"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _Tensor_core.py
    Description:

    Created by Melrose-Lbt 2022-2-28
"""
from _Tensor_core import Tensor
import numpy as np


# TODO: Test operation function from outside project
def add(tensor1, tensor2):
    """
        Class method for adding operation.
        Examples:
            Tensor.add(a, b)

        :return an instantiated Tensor
    """
    return Tensor(np.add(tensor1.value, tensor2.value))


def mul(tensor1, tensor2):
    """
        Element-wise multiply two tensors.
        Examples:
            Tensor.mul(a, b)

        :return an instantiated Tensor
    """
    return Tensor(np.multiply(tensor1.value, tensor2.value))


def matmul(tensor1, tensor2):
    """
        Matrix multiply.
        Examples:
            Tensor.matmul(a, b)

        :return an instantiated Tensor
    """
    return Tensor(np.matmul(tensor1.value, tensor2.value))


def exp(tensor):
    """
        Exponential operation.
        Examples:
            Tensor.exp(a)

        :return an instantiated Tensor
    """
    return Tensor(np.exp(tensor.value))


def sum(tensor):
    """
        Add tensor's value all together.
        :return: an instantiated Tensor
    """
    return Tensor(tensor.value.sum())
