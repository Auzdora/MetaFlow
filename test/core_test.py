"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: core_test.py
    Description: This file basically used for testing core elements during coding
        process.

    Created by Melrose-Lbt 2022-3-17
"""
from core import Tensor, F
from core import Sigmoid, Sum, MatMul, Exp, Mul, Add, SoftMax
from layers import Conv2D
import numpy as np
from loss_fn import LossMSE, CrossEntropy

# label = [[0, 1, 0, 0], [0, 0, 1, 0]]
# a = Tensor([[[1], [2], [1]], [[1], [1], [1]]])
# c = Tensor([[[2, 2, 1], [2, 1, 0], [-1, 1, 1], [-2, 1, 2]]], grad_require=True)
# s = Tensor([[0], [1], [1], [-1]], grad_require=True)
# d = MatMul(c, a)
# l = Add(d, s)
# o = SoftMax(l)
# loss = LossMSE(label, o)
# loss.backward()
# print(loss)
# print(o.grad)
# print(c.grad)

img = Tensor((1, 1, 255, 255))
conv = Conv2D(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=3, _bias=True)
output = conv(img)




