"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _Operators.py
    Description:

    Created by Melrose-Lbt 2022-2-28
"""
import dask.rewrite
import numpy as np
import utils
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
        tensors = self.connect_tensor(*args)
        self.grad_fn = "add"

        return np.add(*tensors)

    def compute_jacobi(self, parent):
        return np.array(np.eye(self.shape[0]))


class Mul(Operator):
    """
        Mul operator.
    """
    def compute_value(self, *args):
        # Define relationship
        tensors = self.connect_tensor(*args)
        self.grad_fn = "multiply"

        return np.multiply(*tensors)

    def compute_jacobi(self, parent):
        pass


class MatMul(Operator):
    """
        Matrix wise operator.
    """
    def compute_value(self, *args):
        # Define relationship
        tensors = self.connect_tensor(*args)
        self.grad_fn = "matrix multiply"

        return np.matmul(*tensors)

    def compute_jacobi(self, parent):
        # If self.shape is one dimensional vector, choose this
        # TODO: Add doc, explain this code blocks
        if parent is self.parents[0]:
            other_parent = self.parents[1]
        else:
            other_parent = self.parents[0]
        container = np.zeros((self.shape[0]*self.shape[1], parent.shape[0]*parent.shape[1]))
        return utils.renew_to_diag(container, other_parent)


if __name__ == "__main__":
    a = Tensor([0, 1])
    print(a)
    b = Tensor([[4, 1], [7, 6]], grad_require=True)
    c = MatMul(b, a)
    c.backward()
    print(c.grad)
    print(b.grad)

