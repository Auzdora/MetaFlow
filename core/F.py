"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: F.py
    Description: Define forward propagation function.

    Created by Melrose-Lbt 2022-3-6
"""
from numpy import ndarray

from core import Add, MatMul, Sigmoid, Conv2D, Tensor
import numpy as np


def fully_connected_layer(input, weight, bias):
    """
        Compute logic of fully connected layer.
    :param input: Input Tensor
    :param weight: weight parameters
    :param bias: bias parameters
    :return:
    """
    assert weight.shape[2] == input.shape[1], "Weight matrix col:{} and input matrix row:{} conflict! " \
        .format(weight.shape[2], input.shape[1])
    if bias is None:
        return MatMul(weight, input)
    else:
        assert bias.shape[0] == weight.shape[1], "Bias matrix row:{} and MatMul(weight, input) matrix row:{} conflict!"\
        .format(bias.shape[0], weight.shape[1])
        return Add(MatMul(weight, input), bias)


def sigmoid(input):
    """
        Sigmoid function is g(x) = 1/(1 + exp(-x)), this function packs sigmoid
    operator defined in '_Operators' ( because it needs to calculate gradent when
    backpropagation.
    :param input:
    :return:
    """
    return Sigmoid(input)


def convolution_2d(input, kernels, **kwargs):
    """
        Packing useful parameters all together into kwargs.
        Padding input Tensor based on 'padding' params.

        ---------------------------------------------------
        :param input: input Tensor
        :param kernels: conv kernels
        :param bias: kernels' bias
        :param stride: step
        :param padding: 0 padding number
    """
    padding = kwargs['padding']
    input.value = zero_padding(input.value, padding_size=padding)
    input.shape = input.value.shape
    return Conv2D(input, kernels, **kwargs)


def zero_padding(in_array, padding_size=0):
    """
        Zero padding function for convolution operation.
        in_array is input image, padding_size will fill zeros around input image,
    for each channel and each batch.
    """
    assert isinstance(in_array, ndarray), "argument 1 should be ndarray"
    assert len(in_array.shape) == 4, "input data should be 4 dimension, (B, C, H, W) for each dim"

    batch, channels, rows, cols = in_array.shape
    padding_array = np.zeros((batch, channels, rows + 2 * padding_size, cols + 2 * padding_size))
    padding_array[:, :, padding_size:rows + padding_size, padding_size:cols + padding_size] = in_array

    return padding_array


def maxpooling(input):

    return Max