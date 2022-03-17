"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: core_test.py
    Description: This file basically used for testing core elements during coding
        process.

    Created by Melrose-Lbt 2022-3-17
"""
from core import Tensor
from core import Sigmoid, Sum, MatMul, Exp

a = Tensor([1, 2, 3])
c = Tensor([[1, 2, 1], [2, 1, 0]], grad_require=True)
d = MatMul(c, a)
b = Exp(d)
print(b)
b.backward()
print(c.grad)
