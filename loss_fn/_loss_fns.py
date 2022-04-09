"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _loss_fns.py
    Description: Different kinds of loss functions are defined here.

    Created by Melrose-Lbt 2022-3-5
"""
from numpy import ndarray

from core import Tensor
import numpy as np


class BaseLoss(Tensor):
    """
        'Rename' Tensor class to Operator.
    """
    def compute_value(self, *args):
        pass

    def compute_jacobi(self, *args):
        pass


class LossMSE(BaseLoss):
    """
        Mean square error loss function.
    """
    def __init__(self, label, outputs):

        if isinstance(label, ndarray):
            pass
        else:
            label = np.array(label)

        if len(label.shape) == 1:
            self.label = np.expand_dims(label, axis=0)
        else:
            self.label = label.T

        # Number of samples
        self.batch = len(label)

        self.jacobi_coef = 1
        super(LossMSE, self).__init__(*[outputs], grad_fn='<LossMSE>', special_op=True)

    def compute_value(self, *args):
        # TODO: Add assert to make sure label dim equals to output dim
        outputs = self.connect_tensor(args[0])
        _tSummer = 0
        for index, output in enumerate(outputs[0]):
            _tSummer += ((output - np.expand_dims(self.label[:, index], axis=1)) ** 2).mean()
        return _tSummer/self.batch

    def compute_jacobi(self, parent):
        # TODO:Explain here why we need add .T
        _tCounter, _tSummer = 0, 0
        for label_index, _subTensor in enumerate(parent.value):
            _tSummer += self.jacobi_coef * (_subTensor - np.expand_dims(self.label[:, label_index], axis=1)).T
            _tCounter += 1
        return _tSummer/_tCounter


class CrossEntropy(BaseLoss):
    """
        Cross entropy loss function.
    """
    def __init__(self, label, outputs):
        self.label = np.expand_dims(label, axis=1)

        # Number of samples
        self.N = len(label)
        self.jacobi_coef = 2 / self.N
        super(CrossEntropy, self).__init__(*[outputs], grad_fn='<CrossEntropyLoss>', special_op=True)

    def compute_value(self, *args):
        pass

    def compute_jacobi(self, *args):
        pass
