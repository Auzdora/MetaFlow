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
        super(Sum, self).__init__(tensor, grad_fn='<TensorSum>', special_op=True, grad_require=True)

    def compute_value(self, *args):
        tensors = self.connect_tensor(args[0])
        return np.array(tensors[0]).sum()

    def compute_jacobi(self, parent):
        return np.ones((1, parent.shape[1] * parent.shape[2]))


class Exp(Operator):
    """
        Exponential operator.
    """
    def __init__(self, tensor):
        super(Exp, self).__init__(tensor, grad_fn='<TensorExp>', special_op=True, grad_require=True)

    def compute_value(self, *args):
        tensors = self.connect_tensor(args[0])
        return np.exp(np.array(tensors[0]))

    def compute_jacobi(self, parent):
        _tCounter, _tSummer = 0, 0
        for _subTensor in parent.value:
            print(_subTensor)
            exp_val = np.exp(parent.value).flatten()
            _tSummer += np.diag(exp_val)
            _tCounter += 1

        return _tSummer/_tCounter


class Sigmoid(Operator):
    """
        Sigmoid: g(x) = 1 / (1 + exp(-x))
    """
    def __init__(self, tensor):
        super(Sigmoid, self).__init__(tensor, grad_fn='<TensorSigmoid>', special_op=True, grad_require=True)

    def compute_value(self, *args):
        tensors = self.connect_tensor(args[0])
        e = 1 / np.exp(tensors[0])
        return 1 / (1 + e)

    def compute_jacobi(self, parent):
        """
            g'(x) = g(x) (1 - g(x))
        """
        _tCounter, _tSummer = 0, 0
        for _subTensor in parent.value:
            e = 1 / np.exp(_subTensor)
            g = np.array(1 / (1 + e))
            d_sig = (g * (1 - g)).flatten()
            _tSummer += np.diag(d_sig)
            _tCounter += 1
        return _tSummer/_tCounter


class Add(Operator):
    """
        Add operator.
    """
    def __init__(self, *args):
        super(Add, self).__init__(*args, grad_fn='<TensorAdd>', grad_require=True)

    def compute_value(self, *args):
        # Define relationship
        tensors_list = self.connect_tensor(*args)
        return np.add(*tensors_list)

    def compute_jacobi(self, parent):
        return np.array(np.eye(self.shape[1] * self.shape[2]))


class Minus(Operator):
    """
        Minus operator.
    """
    def __init__(self, *args):
        super(Minus, self).__init__(*args, grad_fn='<TensorMinus>', grad_require=True)

    def compute_value(self, *args):
        # Define relationship
        tensors = self.connect_tensor(*args)
        # TODO: np.add??? For minus??
        return np.add(*tensors)

    def compute_jacobi(self, parent):
        return np.array(np.eye(self.shape[1] * self.shape[2]))


class Mul(Operator):
    """
        Mul operator.
    """
    def __init__(self, *args):
        super(Mul, self).__init__(*args, grad_fn='<TensorMul>', grad_require=True)

    def compute_value(self, *args):
        # Define relationship
        tensors = self.connect_tensor(*args)
        self.grad_fn = '<TensorMul>'

        return np.multiply(*tensors)

    def compute_jacobi(self, parent):
        if parent is self.parents[0]:
            other_parent = self.parents[1]
        else:
            other_parent = self.parents[0]
        pass


class MatMul(Operator):
    """
        Matrix wise operator.
        UseWarning: When you use this operator, you have to pay attention tensors order. Because it has two
            different ways to compute jacobi matrix with respect to the order of tensor.
    """
    def __init__(self, *args):
        super(MatMul, self).__init__(*args, grad_fn='<TensorMatMul>', grad_require=True)

    def compute_value(self, *args):
        # Define relationship
        tensors = self.connect_tensor(*args)
        return np.einsum('kij,bjn->bin', *tensors)

    def compute_jacobi(self, parent):
        # If self.shape is one dimensional vector, choose this
        # TODO: Add doc, explain this code blocks
        # Two important parameters
        self_index_num = self.shape[1] * self.shape[2]
        parent_index_num = parent.shape[1] * parent.shape[2]

        # Build container
        container = np.zeros((self_index_num, parent_index_num))

        # Judge logic, Y = W * X
        # dY / dW situation
        if parent is self.parents[0]:
            other_parent = self.parents[1]
            _tCounter, _tSummer = 0, 0

            for _subTensor in other_parent.value:
                _tSummer += utils.renew_to_diag(container, _subTensor)
                _tCounter += 1

            return _tSummer/_tCounter

        # dY / dX situation
        else:
            other_parent = self.parents[0]
            _tCounter, _tSummer = 0, 0

            for _subTensor in other_parent.value:
                jacobi = utils.renew_to_diag(container, _subTensor, w_or_x=False)

                row_order = np.arange(0, self_index_num).reshape(self.shape[2], self.shape[1]).\
                    T.reshape(self_index_num)
                col_order = np.arange(0, parent_index_num).reshape(parent.shape[2], parent.shape[1]).\
                    T.reshape(parent_index_num)

                _tSummer += jacobi[row_order, :][:, col_order]
                _tCounter += 1

            return _tSummer/_tCounter


class SoftMax(Operator):
    def __init__(self, *args):
        super(SoftMax, self).__init__(*args, grad_fn='<TensorSoftmax>', grad_require=True)

    def compute_value(self, *args):
        tensors = self.connect_tensor(*args)
        _e = np.exp(tensors[0])
        p = []
        for index, inputs in enumerate(_e):
            p.append(inputs / np.sum(inputs, axis=0))
        return np.array(p)

    def compute_jacobi(self, parent):
        _tCounter, _tSummer = 0, 0
        for _subTensor in self.value:
            diag = _subTensor.T
            Y = np.diag(diag.squeeze(0))
            Y_2 = np.matmul(_subTensor, _subTensor.T)
            _tSummer += Y - Y_2
            _tCounter += 1
        return _tSummer / _tCounter


class Conv2D(Operator):
    def __init__(self, *args, **kwargs):
        self.padded = args[0].value # input padded image
        self.bias = kwargs['bias']
        self.stride = kwargs['stride']
        self.kernel_size = kwargs['kernel_size']
        self.in_channels = kwargs['in_channels']
        self.out_channels = kwargs['out_channels']

        self.output_h = None
        self.output_w = None

        self.origin_shape = None

        super(Conv2D, self).__init__(*args, grad_fn='<TensorConv>', grad_require=True)

    def compute_value(self, *args):
        tensors = self.connect_tensor(*args)
        _input, _kernel = tensors[0], tensors[1]
        N, C, H, W = _input.shape

        # output shape define
        self.output_h = (H - self.kernel_size) / self.stride + 1
        self.output_w = (W - self.kernel_size) / self.stride + 1

        assert self.output_h % 1 == 0, "output H must be integer"
        assert self.output_w % 1 == 0, "output W must be integer"
        self.output_h = int(self.output_h)
        self.output_w = int(self.output_w)

        _img = self.columize(_input)

        output = np.dot(_img, _kernel.reshape(self.out_channels, -1).transpose(1, 0))

        output += self.bias.value.T
        output = output.reshape((N, self.output_w * self.output_h, self.out_channels)).transpose(0, 2, 1). \
            reshape((N, self.out_channels, self.output_h, self.output_w))

        self.origin_shape = output.shape  # prevent reshape operation destroy the original information
        return output

    def compute_jacobi(self, parent):
        back_images = self.parents[0]
        kernels = self.parents[1]

        N, C, H, W = self.origin_shape
        O, I, Kh, Kw = kernels.shape
        _Kh, _Kw = int(Kh / 2), int(Kw / 2)

        if parent is back_images:
            _tCounter, _tSummer = 0, 0

            # Batch-wise loop
            for sub_kernels in kernels.value:
                # Channel-wise loop
                jacobis = []
                for kernel in sub_kernels:
                    ph, pw = tuple(np.add((H, W), np.multiply((_Kh, _Kw), 2)))
                    _jabobi = []
                    for i in np.arange(_Kw, _Kw + W):
                        for j in np.arange(_Kh, _Kh + H):
                            mask = np.mat(np.zeros((pw, ph)))
                            mask[i - _Kw:i - _Kw + Kw, j - _Kh:j - _Kh + Kh] = kernel
                            _jabobi.append(mask[_Kw:_Kw + W, _Kh:_Kh + H].A1)
                    jacobis.append(np.mat(_jabobi))
                jacobis = np.array(jacobis)
                _tSummer += jacobis
                _tCounter += 1

            return _tSummer / _tCounter

        elif parent is kernels:
            _tCounter, _tSummer = 0, 0

            # Batch-wise loop
            for b_index, sub_images in enumerate(back_images.value):
                # Channel-wise loop
                jacobis = []
                for c_index, image in enumerate(sub_images):
                    _jabobi = []
                    for i in np.arange(_Kw, _Kw + W):
                        for j in np.arange(_Kh, _Kh + H):
                            _jabobi.append(np.mat(self.padded[b_index][c_index][i - _Kw:i - _Kw + Kw, \
                                                  j - _Kh:j - _Kh + Kh]).A1)
                    jacobis.append(np.mat(_jabobi))
                jacobis = np.array(jacobis).repeat(O, axis=1)  # output kernels number
                _tSummer += jacobis
                _tCounter += 1

            return _tSummer / _tCounter



    def columize(self, image):
        """
            Columize the input high-dimensional image to a 2D matrix.
        """
        N, C, H, W = image.shape
        cols = []
        for i in range(0, H - self.kernel_size + 1, self.stride):
            for j in range(0, W - self.kernel_size + 1, self.stride):
                col = image[:, :, i:i + self.kernel_size, j:j + self.kernel_size].reshape(N, -1)
                cols.append(col)
        columize_img = np.array(cols)

        return columize_img.transpose((1, 0, 2)).reshape(-1, C * self.kernel_size * self.kernel_size)


if __name__ == "__main__":
    pass
