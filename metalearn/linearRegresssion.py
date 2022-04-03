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
from core import Tensor, Modules

from layers import Linear
from loss_fn import LossMSE
from opt import SGD, BGD, MiniBGD
from data import DataLoader, Dataset


class LinearData(Dataset):
    def __init__(self, dataset):
        if isinstance(dataset, ndarray):
            self.data = dataset
        else:
            self.data = np.array(dataset)
        self.dims = self.data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_data = np.expand_dims(self.data[:, :-1], axis=-1)
        label = self.data[:, -1]
        return input_data[index], label[index]


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
        self.dataset = LinearData(dataset)
        # dataset params
        self.data_num = len(self.dataset)
        self.data_dim = self.dataset.dims

        self.LinearLoader = DataLoader(self.dataset, 1)
        self.normalization = normalization

        if self.normalization:
            self.mean_container, self.std_container = self._normalize()
        else:
            pass

        # model
        self.model = LinearModel(in_features=self.data_dim-1, out_features=1)
        if str.lower(opt) == 'sgd':
            self.optimizer = SGD(self.model, lr)
        super(LinearRegression, self).__init__()

    def _normalize(self):
        """
            Normalize data.
            Highly recommended if you are using gradient descent method to opt-
        imize the model, because its result could be more accurate.
        """
        mean_container, std_container = [], []
        for index in range(self.data_dim):
            mean = np.mean(self.dataset[:, index])
            std = np.std(self.dataset[:, index])
            self.dataset[:, index] = (self.dataset[:, index]-mean)/std
            mean_container.append(mean)
            std_container.append(std)
        return mean_container, std_container

    def get_model_info(self):
        """
            Call this method then you could get this model's information
        """
        self.model.get_model_info()

    def train(self, iteration_num):
        """
            Train logic, based on gradient descent method.
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
        if self.normalization:
            # duplicate mean and std matrix, -1 represents for labels
            mean = np.tile(self.mean_container[:-1], (data.shape[0], 1))
            std = np.tile(self.std_container[:-1], (data.shape[0], 1))
            data = (data - mean)/std
        output = []
        for input in data:
            output.append(float(self.model(input).value))
        if self.normalization:
            out_mean = np.tile(self.mean_container[-1], (1, data.shape[0])).squeeze(0)
            out_std = np.tile(self.std_container[-1], (1, data.shape[0])).squeeze(0)
            output = np.array(output)
            output = np.multiply(output, out_std) + out_mean
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


class NormalEquation:
    """
        Normal equation method to solve linear regression problem.
    You could use this method when the scale of dataset is small.
    But when you have amount of data to regress, it is better to use
    LinearRegression (Gradient Descent) because normal equation needs
    to take a lot of time and computation energy.
    """
    def __init__(self, dataset):
        self.dataset = np.array(dataset)
        one = np.ones(dataset.shape[0])
        # build X matrix
        self.X = np.column_stack((dataset[:, :-1], one))
        # build Y matrix
        self.Y = dataset[:, -1]
        self.theta = self._compute()

    def _compute(self):
        """
            Optimize solution: theta = (X^T * X)^(-1) * X^T * Y
        :return: theta parameters
        """
        params = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.X.T, self.X)), self.X.T), self.Y)
        return params

    def predict(self, data):
        """
            Predict method
        :param data: list or numpy data
        :return:
        """
        data = np.array(data)
        one = np.ones((data.shape[0]))
        data = np.column_stack((data, one))
        output = np.sum(self.theta * data, axis=1)
        return output