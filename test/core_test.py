"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: core_test.py
    Description: This file basically used for testing core elements during coding
        process.

    Created by Melrose-Lbt 2022-3-17
"""
from core import Tensor
from core import Sigmoid, Sum, MatMul, Exp, Mul, Add, SoftMax
import numpy as np
from loss_fn import LossMSE

label = [[2, 1], [1, 3]]
a = Tensor([[[1], [2], [1]], [[1], [1], [1]]])
c = Tensor([[[2, 2, 1], [2, 1, 0]]], grad_require=True)
s = Tensor([[0], [1]], grad_require=True)
d = MatMul(c, a)
l = Add(d, s)
o = SoftMax(l)
loss = LossMSE(label, o)
loss.backward()
print(o)
print(loss)
print(o.grad)



