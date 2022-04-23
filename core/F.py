"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: F.py
    Description: Define forward propagation function.

    Created by Melrose-Lbt 2022-3-6
"""
from core import Add, MatMul, Sigmoid, Conv2D


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


def convolution_2d(input, kernels, bias, stride, padding):
    """
    :param input:
    :return:
    """
    kwargs = {
        'bias': bias,
        'stride': stride,
        'padding': padding
    }
    return Conv2D(input, kernels, **kwargs)
