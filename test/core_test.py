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
from layers import Conv2D, Linear
import numpy as np
from loss_fn import LossMSE, CrossEntropy

label = [[0, 1, 0, 0], [0, 0, 1, 0]]
linear = Linear(in_features=16, out_features=4)
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

img = Tensor([[[[0.1,0,0.2,0.1], [0.2, 0.7, 0.1, 0.9], [0.2,0.2,0.3,0.1],[0.9,0,0.1,0.3]],
                     [[0.1,0.1,0.4,0], [0.1, 0.3, 0.4, 0.5], [0.2, 0.1, 0.7, 0.3],[0.1, 0.1, 0.1 ,0.1]]]

                    ,[[[0.1,0,0.2,0.1], [0.2, 0.7, 0.1, 0.9], [0.2,0.2,0.3,0.1],[0.9,0,0.1,0.3]],
                     [[0.1,0.1,0.4,0], [0.1, 0.3, 0.4, 0.5], [0.2, 0.1, 0.7, 0.3],[0.1, 0.1, 0.1 ,0.1]]]])
conv1 = Conv2D(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1, _bias=True)
conv2 = Conv2D(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, _bias=True)
x1 = conv1(img)
print('conv1:{}'.format(x1))
x = conv2(x1)
print('conv2:{}'.format(x.shape))
x.value = x.value.reshape(2, -1, 1)
x.shape = x.value.shape
print('reshape:{}'.format(x.shape))
lx = linear(x)
print('linear shape:{}'.format(lx.shape))
out = SoftMax(lx)
print('softmax out:{}{}'.format(out, out.shape))
loss = CrossEntropy(label, out)

loss.backward()
print(loss)






