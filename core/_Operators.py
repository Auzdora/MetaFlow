"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _Operators.py
    Description:

    Created by Melrose-Lbt 2022-2-28
"""
import dask.rewrite
import numpy as np

from _Tensor_core import Tensor


class Operator(Tensor):
    """
        'Rename' Tensor class to Operator.
    """
    def compute_grad(self):
        pass

    def compute_jacobi(self, *args):
        pass


class Add(Operator):
    """
        Add operator.
    """
    def compute_value(self, *args):
        # Define relationship
        self.relationship(*args)
        self.grad_fn = "add"

        # Compute
        tensors = []
        for arg in args:
            tensors.append(arg.value)
        return np.add(*tensors)

    def compute_jacobi(self):
        return np.mat(np.eye(self.shape))


class Mul(Operator):
    """
        Mul operator.
    """
    def compute_value(self, *args):
        # Define relationship
        self.relationship(*args)
        self.grad_fn = "multiply"

        # Compute
        tensors = []
        for arg in args:
            tensors.append(arg.value)
        return np.multiply(*tensors)
        pass

    def compute_jacobi(self):
        pass


if __name__ == "__main__":
    a = Tensor([[1,2,3],[3,2,1]])
    b = Tensor([[2,3,4],[2,3,4]])
    k = Tensor([1,1,1])
    c = Add(a,b)
    print(np.mat(np.eye(c.shape[1])))

