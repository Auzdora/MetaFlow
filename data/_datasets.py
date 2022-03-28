"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _datasets.py
    Description: _datasets.py provides an abstract class for every kind of dataset.
        You could create your own dataset by inheriting from Dataset class. But you
        have to override __getitem__ and __len__ method to make sure that everything
        is all set.
            When you need to create a dataset defined by you self, and you wanna use
        MetaFlow as a framework and use MetaFlow.data.DataLoader to load your data,
        you have to use this class to pack your dataset. Then you could deliver it to
        DataLoader.

    Created by Melrose-Lbt 2022-3-28
"""


class Dataset:
    """
        An abstract class, base class for all Dataset subclass.


        For examples:
            class Mydata(Dataset):
                def __init__(self, dataset):
                    self.dataset = dataset

                def __getitem__(self, index):
                    return self.dataset[index]

                def __len__(self):
                    return len(self.dataset)
    """
    def __getitem__(self, index):
        """
            When a subclass inherits from Dataset, you have to override this method.
            This method is an iterator.
            You have to make sure it returns inputs(data) and targets(label).

        """
        raise NotImplementedError

    def __len__(self):
        """
            When a subclass inherits from Dataset, you have to override this method.
        """
        raise NotImplementedError
