"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: subsample_layers.py
    Description: Contains max pooling layer, average pooling layer and so on.

    Created by Melrose-Lbt 2022-5-10
"""
import numpy as np
from core import Tensor, Modules, F


class MaxPooling(Modules):
    def __init__(self):
        super(MaxPooling, self).__init__(self.core_module)

    def forward(self, x):
        if isinstance(x, Tensor):
            pass

    def get_model_info(self):
        pass

    def reset_parameters(self):
        pass
