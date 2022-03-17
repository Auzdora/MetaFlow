"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _loss_fns.py
    Description: Different kinds of loss functions are defined here.

    Created by Melrose-Lbt 2022-3-5
"""
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
        self.label = np.expand_dims(label, axis=1)

        # Number of samples
        self.N = len(label)
        self.jacobi_coef = 2/self.N
        super(LossMSE, self).__init__(*[outputs], grad_fn='<LossMSE>', special_op=True)

    def compute_value(self, *args):
        # TODO: Add assert to make sure label dim equals to output dim
        outputs = self.connect_tensor(args[0])
        return (((outputs[0]-self.label)**2).sum())/(1 * self.N)

    def compute_jacobi(self, parent):
        # TODO:Explain here why we need add .T
        return self.jacobi_coef * (parent.value - self.label).T


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
