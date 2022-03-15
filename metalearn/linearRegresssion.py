"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: linearRegression.py
    Description: This file defines linear regression model, it includes simple
        linear regression, multiple variables linear regression, normal equation
        and so on.

    Created by Melrose-Lbt 2022-3-15
"""
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from core import Tensor

from core import Modules
from layers import Linear
from loss_fn import LossMSE
from opt import SGD, BGD, MiniBGD


class LinearModel(Modules):
    def __init__(self, in_features, out_features):
        self.layer = Linear(in_features=in_features, out_features=out_features)
        super(LinearModel, self).__init__()

    def forward(self, x):
        x = self.layer(x)
        return x


class LinearRegression:
    """
        Linear regression model
    """
    def __init__(self, dataset, opt='sgd', lr=0.01, normalization=False):
        """
        :param dataset:
        :param opt: you could choose 'SGD','BGD','MBGD' for this version.
        """
        if isinstance(dataset, ndarray):
            self.dataset = dataset
        else:
            self.dataset = np.array(dataset)
        # dataset params
        self.data_num = self.dataset.shape[0]
        self.data_dim = self.dataset.shape[1]

        if normalization:
            self.mean_container, self.std_container = self._normalize()
        else:
            pass

        # model
        self.model = LinearModel(in_features=self.data_dim-1, out_features=1)
        if str.lower(opt) == 'sgd':
            self.optimizer = SGD(self.model, lr)
        super(LinearRegression, self).__init__()

    def _normalize(self):
        mean_container, std_container = [], []
        for index in range(self.data_dim):
            mean = np.mean(self.dataset[:, index])
            std = np.std(self.dataset[:, index])
            self.dataset[:, index] = (self.dataset[:, index]-mean)/std
            mean_container.append(mean)
            std_container.append(std)
        return mean_container, std_container

    def get_model_info(self):
        self.model.get_model_info()

    def train(self, iteration_num):
        """
            Train logic
        """
        for epoch in range(iteration_num):
            mean_loss = 0
            for i in range(self.data_num):
                x = Tensor(np.array(self.dataset[i, :-1]).T)
                target = np.expand_dims(np.array(self.dataset[i, -1]), axis=0)
                output = self.model(x)
                loss = LossMSE(target, output)
                loss.backward()
                self.optimizer.update()
                mean_loss += loss.value
                if epoch == iteration_num-1 and i == self.data_num -1:
                    continue
                loss.clear()
            print("epoch{}: loss:{}".format(epoch, mean_loss / self.data_num))

    def predict(self, data):
        data = np.array(data)
        output = []
        for input in data:
            output.append(float(self.model(input).value))
        return output

    def plot_data(self):
        """
            Plot data.
            This image doesn't contains linear model, it only shows
        original data, it could be used for observing data pattern.
        """
        if self.data_dim == 2:
            plt.scatter(self.dataset[:, 0], self.dataset[:, 1])
            plt.show()
        else:
            raise ValueError("plot_data method doesn't support image show above 2 dim yet!")

    def show(self):
        if self.data_dim == 2:
            params = []
            for items in self.model.get_parameters():
                params.append(float(items[1].value))
            x = np.arange(np.min(self.dataset[:, 0])-0.5, np.max(self.dataset[:, 1]+1))
            y = params[0]*x+params[1]
            plt.scatter(self.dataset[:, 0], self.dataset[:, 1], marker='x', color='red')
            plt.plot(x, y)
            plt.show()
        else:
            raise ValueError("plot_data method doesn't support image show above 2 dim yet!")


if __name__ == "__main__":
    dataset = [[0, 0], [0, 1], [2, 0], [3, 3], [4, 4]]
    linear = LinearRegression(dataset)
    linear.get_model_info()
    linear.train(100)
    linear.show()