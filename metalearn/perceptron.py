"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: perceptron.py
    Description: Perceptron algorithm.

    Created by Melrose-Lbt 2022-3-23
"""
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, dataset, feature_dim, lr, option='single'):
        """
            Input data should use 1 and 0 as label
        """
        one = np.ones(np.array(dataset).shape[0])
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
            self.weights = np.array([1,0,1,0])

    def train(self):
        """
            Train logic for one single perceptron.
        """
        acc = 0
        while acc < self.dataset.shape[0]:
            acc = 0
            for index, item in enumerate(self.feature):
                output = np.matmul(self.weights, item.T)
                print(output)
                if output > 0:
                    acc += 1
                    continue
                elif output <= 0:
                    self.weights = self.weights + self.lr * item
                print(self.weights)

    def show_line(self):
        plt.scatter()


if __name__ == "__main__":
    data = [[1,0,1,1], [0,1,1,1], [1,1,0,0],[0,1,0,0]]
    dataset = [[0,0,0,1],[1,0,0,1],[1,0,1,1],[1,1,0,1],[0,0,1,0],[0,1,1,0],[0,1,0,0],[1,1,1,0]]
    percep = Perceptron(data, 3, 1)
    percep.train()