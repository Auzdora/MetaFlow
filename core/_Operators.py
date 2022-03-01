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


class Sum(Operator):
    """
        Sum operator.
    """
    def compute_value(self, *args):
        tensors = self.connect_tensor(args[0])
        self.grad_fn = "sum"

        return np.array(tensors[0]).sum()

    def compute_jacobi(self, parent):
        return np.ones((1, parent.shape[0]))


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


class Minus(Operator):
    """
        Minus operator.
    """
    def compute_value(self, *args):
        # Define relationship
        tensors = self.connect_tensor(*args)
        self.grad_fn = "minus"

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
    x = Tensor([2, 3, 1])
    w = Tensor([[0.2, 0.5, 0.1], [0.1, 0.8, 0]], grad_require=True)
    b = Tensor([1, 1], grad_require=True)
    k = MatMul(w, x)
    output = Add(k, b)
    loss = Sum(output,output)
    loss.backward()
    print(w.grad)



