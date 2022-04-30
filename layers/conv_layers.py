"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: linear_layers.py
    Description: Convolution layers define.

    Created by Melrose-Lbt 2022-4-23
"""
import numpy as np
from core import Tensor, Modules, F


class Conv2D(Modules):
    """
        2 dimensional convolution layers.
        Input data size should be four dimensional, [ Batch, channel, Height, Weight ]
    other input size is unacceptable.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, _bias, init_para='normal'):
        self.core_module = True

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.en_bias = _bias
        self.init_para = init_para

        self.kernels = Tensor((out_channels, in_channels, kernel_size, kernel_size), grad_require=True)
        if self.en_bias:
            self.bias = Tensor((out_channels, 1), grad_require=True)
        else:
            self.bias = None

        self.kwargs = {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'bias': self.bias,
            'padding': self.padding
        }

        self.reset_parameters()

        super(Conv2D, self).__init__(self.core_module)

    def forward(self, x):
        if isinstance(x, Tensor):
            return F.convolution_2d(x, self.kernels, **self.kwargs)
        else:
            x = Tensor(x)
            return F.convolution_2d(x, self.kernels, **self.kwargs)

    def _get_module_info(self):
        pass

    def reset_parameters(self):
        """
            Reset network's parameters. To make network easier to learn.
        :return:
        """
        # TODO: Confirm and pack them into a single file
        if self.init_para == 'normal':
            self.kernels.value = np.random.normal(loc=0., scale=1., size=(self.out_channels, self.in_channels,
                                                                          self.kernel_size, self.kernel_size))
            if self.bias is None:
                pass
            else:
                self.bias.value = np.random.normal(loc=0., scale=1., size=(self.out_channels, 1))
