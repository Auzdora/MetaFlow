"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _datasets.py
    Description: This file provides a demo for MetaFlow.

    Created by Melrose-Lbt 2022-3-28
"""


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
