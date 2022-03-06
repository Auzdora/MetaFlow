"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _Operators.py
    Description: Operators has two different types. Common op and special op.

    Created by Melrose-Lbt 2022-2-28
"""
import abc

import numpy as np
import utils
from ._Tensor_core import Tensor


class Operator(Tensor):
    """
        'Rename' Tensor class to Operator.
    """
    def compute_value(self, *args):
        pass

    def compute_jacobi(self, *args):
        pass


class Sum(Operator):
    """
        Sum operator.
    """
    def __init__(self, tensor):
        # tensor is a <__main__.Tensor object at 0x7f77c0770100>
        super(Sum, self).__init__(tensor, grad_fn='<TensorSum>', special_op=True)

    def compute_value(self, *args):
        tensors = self.connect_tensor(args[0])
        return np.array(tensors[0]).sum()

    def compute_jacobi(self, parent):
        return np.ones((1, parent.shape[0]))


class Add(Operator):
    """
        Add operator.
    """
    def __init__(self, *args):
        super(Add, self).__init__(*args, grad_fn='<TensorAdd>')

    def compute_value(self, *args):
        # Define relationship
        tensors_list = self.connect_tensor(*args)
        return np.add(*tensors_list)

    def compute_jacobi(self, parent):
        return np.array(np.eye(self.shape[0]))


class Minus(Operator):
    """
        Minus operator.
    """
    def __init__(self, *args):
        super(Minus, self).__init__(*args, grad_fn='<TensorMinus>')

    def compute_value(self, *args):
        # Define relationship
        tensors = self.connect_tensor(*args)
        # TODO: np.add??? For minus??
        return np.add(*tensors)

    def compute_jacobi(self, parent):
        return np.array(np.eye(self.shape[0]))


class Mul(Operator):
    """
        Mul operator.
    """
    def __init__(self, *args):
        super(Mul, self).__init__(*args, grad_fn='<TensorMul>')

    def compute_value(self, *args):
        # Define relationship
        tensors = self.connect_tensor(*args)
        self.grad_fn = '<TensorMul>'

        return np.multiply(*tensors)

    def compute_jacobi(self, parent):
        pass


class MatMul(Operator):
    """
        Matrix wise operator.
        UseWarning: When you use this operator, you have to pay attention tensors order. Because it has two
            different ways to compute jacobi matrix with respect to the order of tensor.
    """
    def __init__(self, *args):
        super(MatMul, self).__init__(*args, grad_fn='<TensorMatMul>')

    def compute_value(self, *args):
        # Define relationship
        tensors = self.connect_tensor(*args)
        return np.matmul(*tensors)

    def compute_jacobi(self, parent):
        # If self.shape is one dimensional vector, choose this
        # TODO: Add doc, explain this code blocks
        # Two important parameters
        self_index_num = self.shape[0] * self.shape[1]
        parent_index_num = parent.shape[0] * parent.shape[1]

        # Build container
        container = np.zeros((self_index_num, parent_index_num))

        # Judge logic, Y = W * X
        # dY / dW situation
        if parent is self.parents[0]:
            other_parent = self.parents[1]
            return utils.renew_to_diag(container, other_parent)
        # dY / dX situation
        else:
            other_parent = self.parents[0]
            jacobi = utils.renew_to_diag(container, other_parent, w_or_x=False)
            row_order = np.arange(0, self_index_num).reshape(self.shape[1], self.shape[0]).\
                T.reshape(self_index_num)
            col_order = np.arange(0, parent_index_num).reshape(parent.shape[1], parent.shape[0]).\
                T.reshape(parent_index_num)
            return jacobi[row_order, :][:, col_order]


if __name__ == "__main__":
    pass
