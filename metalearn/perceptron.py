"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: perceptron.py
    Description: Perceptron algorithm.

    Created by Melrose-Lbt 2022-3-23
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Perceptron:
    def __init__(self, dataset, feature_dim, lr, option='single'):
        """
            Input data should use 1 and 0 as label
        """
        one = np.ones(np.array(dataset).shape[0])
        self.origin = np.array(dataset)
        self.dataset = np.insert(dataset, -1, one, axis=1)
        self.dataset[self.dataset[:, -1] == 0] = -self.dataset[self.dataset[:, -1] == 0]
        self.feature_dim = feature_dim
        self.lr = lr
        print(self.dataset)

        # build data and label
        self.feature = self.dataset[:, :-1]
        self.label = self.dataset[:, -1]

        # build weights
        if option == 'single':
            #self.weights = np.random.rand(1, self.feature_dim+1)
            #self.weights = np.zeros((1, self.feature_dim+1))
            self.weights = np.array([-1, -2, -2, 0])

    def train(self):
        """
            Train logic for one single perceptron.
        """
        acc = 0
        while acc < self.dataset.shape[0]:
            acc = 0
            for index, item in enumerate(self.feature):
                output = np.matmul(self.weights, item.T)
                if output > 0:
                    acc += 1
                    continue
                elif output <= 0:
                    self.weights = self.weights + self.lr * item
                print(self.weights)

    def show(self):
        assert self.feature_dim <= 3, "show method can't show image above 3D"
        if self.feature_dim == 2:
            if self.weights[0] == 0:
                x = np.linspace(-2, 2, 100)
                y = self.weights[0]*x/self.weights[1] - self.weights[2]/self.weights[1]
                plt.plot(x, y)
                print(self.origin[:, 0], self.origin[:, 1])
                plt.scatter(self.origin[:, 0], self.origin[:, 1])
            elif self.weights[1] == 0:
                y = np.linspace(-2, 2, 100)
                x = self.weights[1]*y/self.weights[0] - self.weights[2]/self.weights[0]
                plt.plot(x, y)
                print(self.origin[:, 0], self.origin[:, 1])
                plt.scatter(self.origin[:, 0], self.origin[:, 1])
            else:
                x1 = np.linspace(-2, 2, 100)
                x2 = self.weights[0]*x1/self.weights[1] + self.weights[2]/self.weights[1]
                plt.plot(x1, x2)
                plt.scatter(self.origin[:, 0], self.origin[:, 1])
            plt.show()
        elif self.feature_dim == 3:
            ax = plt.figure().add_subplot(111, projection='3d')
            ax.scatter(self.origin[:, 0], self.origin[:, 1], self.origin[:, 2], c='r', marker='o')
            x = np.linspace(-1, 1.5, 100)
            y = np.linspace(-1, 1.5, 100)
            X, Y = np.meshgrid(x, y)
            ax.plot_surface(X,
                            Y,
                            Z=self.weights[0]*X/(-self.weights[2]) + self.weights[1]*X/(-self.weights[2])\
                              + self.weights[3]/(-self.weights[2]),
                            color='b')
            plt.show()


if __name__ == "__main__":
    data = [[0,1,1], [1,1,1], [0,0,0], [1,0,0]]
    dataset = [[0,0,0,1],[1,0,0,1],[1,0,1,1],[1,1,0,1],[0,0,1,0],[0,1,1,0],[0,1,0,0],[1,1,1,0]]
    percep = Perceptron(dataset, 3, 1)
    percep.train()
    percep.show()