"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _Operators.py
    Description: Operators has two different types. Common op and special op.

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
    def __init__(self, tensor):
        args = [tensor]
        super(Sum, self).__init__(*args, special_op=True)

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
        UseWarning: When you use this operator, you have to pay attention tensors order. Because it has two
            different ways to compute jacobi matrix with respect to the order of tensor.
    """
    def compute_value(self, *args):
        # Define relationship
        tensors = self.connect_tensor(*args)
        self.grad_fn = "matrix multiply"

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
            a = jacobi[row_order, :][:, col_order]
            return jacobi[row_order, :][:, col_order]


class LossMSE(Operator):
    """
        Mean square error loss function.
    """
    def __init__(self, label, outputs):
        self.label = np.expand_dims(label,axis=1)
        self.output = [outputs]
        # Number of samples
        self.N = len(label)
        self.jacobi_coef = 2/self.N
        super(LossMSE, self).__init__(*self.output, special_op=True)

    def compute_value(self, *args):
        # TODO: Add assert to make sure label dim equals to output dim
        outputs = self.connect_tensor(args[0])
        self.grad_fn = "mse"
        return (((outputs[0]-self.label)**2).sum())/self.N

    def compute_jacobi(self, parent):
        # TODO:Explain here why we need add .T
        return self.jacobi_coef * (parent.value - self.label).T


if __name__ == "__main__":
    import time
    past = time.time()
    x = Tensor([2, 3, 1])
    w1 = Tensor([[0.2, -0.1, 0.1], [-0.12, 0.05, 0.3]], grad_require=True)
    b1 = Tensor([1, 1], grad_require=True)
    w2 = Tensor([[0.1, 0.2], [0.1, -0.1]], grad_require=True)
    b2 = Tensor([1, 1], grad_require=True)
    label = np.array([7, 9])
    for epoch in range(20):
        #output = Add(MatMul(w2, Add(MatMul(w1, x), b1)), b2)
        n = MatMul(w1, x)
        z = Add(n, b1)
        c = MatMul(w2, z)
        #output = MatMul(w2, Add(MatMul(w1, x), b1))
        loss = LossMSE(label, c)
        loss.backward()
        w1.value = w1.value - 0.01 * w1.grad.reshape(2,3)
        w2.value = w2.value - 0.01 * w2.grad.reshape(2,2)
        b1.value = b1.value - 0.01 * b1.grad.reshape(2,1)
        #b2.value = b2.value - 0.01 * b2.grad.reshape(2,1)
        print("epoch{}: loss:{}".format(epoch, loss))
    now = time.time()
    print("run time:{}s".format(now-past))


