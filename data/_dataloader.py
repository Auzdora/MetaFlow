"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _dataloader.py
    Description: This file provides a demo for MetaFlow.

    Created by Melrose-Lbt 2022-3-28
"""
import numpy as np


class DataLoader:
    def __init__(self, datasets, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        pass

    def __next__(self):
        pass
