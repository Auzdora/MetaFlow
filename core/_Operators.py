"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _Operators.py
    Description:

    Created by Melrose-Lbt 2022-2-28
"""
import dask.rewrite
import numpy as np

from _Tensor_core import Tensor


class Add(Tensor):
    def __init__(self, *args):
        super(Add, self).__init__(*args)

    def compute_value(self, *args):
        tensors = []
        for arg in args:
            tensors.append(arg.value)
        new_tensor = Tensor(np.add(*tensors))
        # Define relationship
        for tensor in args:
            tensor.children.append(new_tensor)
            self.parents.append(tensor)

        return new_tensor

    def compute_grad(self):
        pass


if __name__ == "__main__":
    a = Tensor([1,2,3])
    b = Tensor([2,3,4])
    k = Tensor([1,1,1])
    c = Add(a, b)
    m = Add(c,k)
    print(m.get_parents())

