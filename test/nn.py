import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from loss_fn import CrossEntropy, LossMSE
from layers import Sigmoid, Linear
from core import Modules, Tensor
from data import DataLoader, Dataset
from opt import SGD
from matplotlib import pyplot as plt

data = loadmat('nn_data/ex4data1.mat')
X = data['X']
y = data['y']
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)


class Model(Modules):
    def __init__(self):
        self.layer1 = Linear(in_features=2, out_features=1)
        self.sig = Sigmoid()
        super(Model, self).__init__()

    def forward(self, x):
        x = self.layer1(x)
        x = self.sig(x)

        return x