"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: core_test.py
    Description: This file basically used for testing core elements during coding
        process.

    Created by Melrose-Lbt 2022-3-17
"""
from core import Tensor
from core import Sigmoid, Sum, MatMul, Exp, Mul, Add

a = Tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
c = Tensor([[0.1, 0.2, 0.1], [0.2, 0.1, 0]], grad_require=True)
s = Tensor([[1, 1], [1, 1]], grad_require=True)
d = MatMul(c, a)
l = Add(d, s)
b = Exp(l)
m = Sigmoid(b)
print(m)
k = Sum(m)
print(k)
k.backward()
print(c.grad)
c.value = c.value - 0.01 * c.grad.reshape(2, 3)
