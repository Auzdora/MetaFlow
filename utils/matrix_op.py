"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: matrix_op.py
    Description: Define some of useful tools for matrix operation.

    Created by Melrose-Lbt 2022-3-1
"""
import numpy as np


def renew_to_diag(container, parent_tensor, w_or_x=True):
    """
        This function provides a method for ./core/_Operatiors.py/MatMul.compute_jacobi.
    It takes a created container which full of zeros and a parent Tensor. And then refresh
    container to a brand new diag container.
    :param w_or_x: weight or x -> bool
    :param container: A zero array
    :param parent_tensor: parent Tensor
    :return: a renewed container
    """
    # Get important parameters
    if w_or_x:
        # If w_or_x is true, that means we are compute dY/dW where Y=WX
        value_t = parent_tensor.value.T
    else:
        # If w_or_x is false, that means we are compute dY/dX where Y=WX
        value_t = parent_tensor.value
    row_step = value_t.shape[0]
    col_step = value_t.shape[1]
    iter_num = container.shape[0]/row_step

    # Renew container
    for index in range(int(iter_num)):
        container[index*row_step: index*row_step+row_step, index*col_step:index*col_step+col_step] = value_t
    return container


if __name__ == "__main__":
    pass
