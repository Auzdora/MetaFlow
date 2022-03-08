"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: distance.py
    Description: This file contains distance calculation methods.

    Created by Melrose-Lbt 2022-3-7
"""
import numpy as np
from numpy import ndarray
from core import Tensor
import math


def euclidean_distance(tensor1, tensor2):
    assert type(tensor1) == type(tensor2), "tensor1 type:{} is not same as tensor2 type:{}"\
        .format(type(tensor1), type(tensor2))

    if isinstance(tensor1, ndarray):
        assert tensor1.shape == tensor2.shape, "tensor1 shape:{} is not aligned with tensor2 shape:{}"\
            .format(tensor1.shape, tensor2.shape)
        return np.sqrt(((tensor1 - tensor2)**2).sum())

    elif isinstance(tensor1, Tensor):
        assert tensor1.shape == tensor2.shape, "tensor1 shape:{} is not aligned with tensor2 shape:{}" \
            .format(tensor1.shape, tensor2.shape)
        return Tensor(np.sqrt(((tensor1.value - tensor2.value)**2).sum()))

    elif isinstance(tensor1, list):
        assert len(tensor1) == len(tensor2), "tensor1 shape:{} is not aligned with tensor2 shape:{}" \
            .format(len(tensor1), len(tensor2))
        return np.sqrt(((np.array(tensor1) - np.array(tensor2))**2).sum())

    else:
        origin_type = type(tensor1)
        return origin_type(np.sqrt(((tensor1 - tensor2)**2)))
