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
    def __init__(self, dataset, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._counter = 0

        if self.shuffle:
            self.dataset = dataset
            np.random.shuffle(self.dataset.data)
        else:
            self.dataset = dataset

    def __iter__(self):
        return self

    def __next__(self):
        if self._counter * self.batch_size >= len(self.dataset):
            self._counter = 0
            raise StopIteration
        else:
            input_batch = []
            target_batch = []
            for index in range(self.batch_size):
                _indexor = self._counter * self.batch_size + index
                if _indexor < len(self.dataset):
                    _data, _label = self.dataset[_indexor]
                    input_batch.append(_data)
                    target_batch.append(_label)
                else:
                    break
            self._counter += 1
            return np.array(input_batch), np.array(target_batch)
