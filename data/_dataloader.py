"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _dataloader.py
    Description: _dataloader.py is a universal controller that controls data input to your
        self-defined network. It could collocate your data to multiple mini-batch, which
        could accelerate calculation speed. Furthermore, it could let networks learn more
        useful information. It could also shuffle your original dataset to make sure that
        networks don't learn something that is unnecessary.

    Created by Melrose-Lbt 2022-3-28
"""
import numpy as np


class DataLoader:
    """
        DataLoader, data loader, an abstract class.
        Base class for all dataloader subclasses.
        If you want to use this class, you have write a subclass and let it inherits from
        DataLoader.
    """
    def __init__(self, datasets, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        pass

    def __next__(self):
        pass
