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
            # one label 0 or 1
            self.N = 1
            self.label = np.expand_dims(label, axis=0)
        else:
            # multi [0, 0, 1, ..., 0] one hot label
            # if it is batch, it has to construct like this shape    [batch, label_dim]
            self.N = label.shape[1]
            self.label = label.T

        self.jacobi_coef = 2/self.N
        super(LossMSE, self).__init__(*[outputs], grad_fn='<LossMSE>', special_op=True)

    def compute_value(self, *args):
        # TODO: Add assert to make sure label dim equals to output dim
        outputs = self.connect_tensor(args[0])
        _tSummer, _tCounter = 0, 0
        for index, output in enumerate(outputs[0]):
            _tSummer += ((output - np.expand_dims(self.label[:, index], axis=1)) ** 2).mean()
            _tCounter += 1
        return _tSummer/_tCounter

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
        if isinstance(label, ndarray):
            pass
        else:
            label = np.array(label)
        self.label = label

        super(CrossEntropy, self).__init__(*[outputs], grad_fn='<CrossEntropyLoss>', special_op=True)

    def compute_value(self, *args):
        outputs = self.connect_tensor(args[0])
        _tSummer, _tCounter = 0, 0
        for index, _one_batch_data in enumerate(outputs[0]):
            _tSummer += - np.sum(np.multiply(self.label[index], np.log(_one_batch_data).squeeze(1)))
            _tCounter += 1
        return _tSummer / _tCounter

    def compute_jacobi(self, parent):
        _tCounter, _tSummer = 0, 0
        for index, _subTensor in enumerate(parent.value):
            _tSummer += - self.label[index] / _subTensor.squeeze(1)
            _tCounter += 1
        return np.expand_dims(_tSummer, axis=0) / _tCounter
