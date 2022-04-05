"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: _Optimizers.py
    Description: This file provides different types of optimizers.

    Created by Melrose-Lbt 2022-3-5
"""
import abc

import numpy as np

from core import Modules

beta_1 = 0.9  # momentum decay ratio / RMSProp decay ratio / Adam decay ratio_1
beta_2 = 0.99  # Adam decay ratio_2
Epsilon = 1e-10  # infinite min value


class Optimizer(abc.ABC):
    """
        Abstract class for optimizers.
    """

    def __init__(self, model, learning_rate=0.001):
        if learning_rate < 0.0:
            raise ValueError("learning rate value should be positive value.")
        assert isinstance(model, Modules), \
            "input model is not a Modules class."
        # Access to all the leaf nodes that are gradable.
        # Number of gradable leaf nodes.
        self.model = model
        self.learning_rate = learning_rate

    @abc.abstractmethod
    def update(self):
        """
            Params update method, need to be implemented when you create a
        sub class. Gradient descent is baseline.
        """
        raise NotImplementedError("optimizer update method undefined, you have to write it.")


class SGD(Optimizer):
    """
        Stochastic Gradient Descent.
    """

    def __init__(self, model, lr=0.001, beta=0.9, epsilon=1e-10, ratio_decay=False, momentum=False,
                 adagrad=False, rmsprop=False):
        # model and learning rate
        self.model = model
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon

        # different optimization method button
        self.ratio_decay = ratio_decay
        self.momentum = momentum
        self.AdaGrad = adagrad
        self.RMSProp = rmsprop

        self.v = dict()  # velocity of momentum
        self.s = dict()  # state of AdaGrad and RMSProp

        super(SGD, self).__init__(model, learning_rate=lr)

    def update(self):
        """
            Stochastic gradient descent optimizer update process. This method congregates naive
        SGD, Momentum, Adagrad and RMSProp method into a single function. At same time it prov-
        ides decay ratio for SGD.
        """
        for p_name, params in self.model.parameters():
            g = params.grad.reshape(params.shape)
            lg = self.lr * g
            if self.AdaGrad or self.RMSProp:
                g_2 = np.power(g, 2)

            if self.momentum:
                if p_name not in self.v:
                    self.v[p_name] = g
                else:
                    self.v[p_name] = self.beta * self.v[p_name] - lg
                params.value = params.value + self.v[p_name]

            elif self.AdaGrad:
                if p_name not in self.s:
                    self.s[p_name] = g_2
                else:
                    self.s[p_name] = self.s[p_name] + g_2
                params.value = params.value - lg / np.sqrt(self.s[p_name] + self.epsilon)
            elif self.RMSProp:
                if p_name not in self.s:
                    self.s[p_name] = g_2
                else:
                    self.s[p_name] = self.beta * self.s[p_name] + (1 - self.beta) * g_2
                params.value = params.value - lg / np.sqrt(self.s[p_name] + self.epsilon)

            else:
                params.value = params.value - lg


class Adam(Optimizer):
    """
        Adam optimizer:
            Adam is an optimization algorithm that can be used instead of the classical stochastic
        gradient descent procedure to update network weights iterative based in training data.

            Adam realizes the benefits of both AdaGrad and RMSProp.Instead of adapting the parameter
        learning rates based on the average first moment (the mean) as in RMSProp, Adam also makes
        use of the average of the second moments of the gradients (the uncentered variance).

            Specifically, the algorithm calculates an exponential moving average of the gradient
        and the squared gradient, and the parameters beta1 and beta2 control the decay rates of
        these moving averages.The initial value of the moving averages and beta1 and beta2 values
        close to 1.0 (recommended) result in a bias of moment estimates towards zero. This bias is
        overcome by first calculating the biased estimates before then calculating bias-corrected
        estimates.The paper is quite readable and I would encourage you to read it if you are
        interested in the specific implementation details.

            In this method, by default, adam parameters are:
            learning rate: 0.001
            beta_1: 0.9
            beta_2: 0.99
            epsilon: 1e-8
            bias correction: False
            weight decay: 0

            TIPs: Bias correction make sure the denominator is not zero at the beginning of training
        process. This correction effect will decay overtime.
            Weight decay is L2 regularization method.
    """
    def __init__(self, model, lr=0.001, betas=(0.9, 0.99), epsilon=1e-8, bias_fix=False, weight_decay=0):
        # TODO: Understand Regularization and add 'weight_decay' to Adam
        self.model = model
        self.lr = lr
        self.epsilon = epsilon
        self.beta_1 = betas[0]
        self.beta_2 = betas[1]
        self.weight_decay = weight_decay

        # iterate counter
        self.t = 0
        # True: it will tune v and s
        self.bias_fix = bias_fix

        self.v = dict()
        self.s = dict()

        super(Adam, self).__init__(model, learning_rate=lr)

    def update(self):
        """
            Update logic. It goes like this:
            1. g = dLoss / dw
            2. g^2 = g * g
            3. v_{t} = beta_1 * v_{t-1} + (1 - beta_1) * g
            4. s_{t} = beta_2 * s_{t-1} + (1 - beta_2) * g
            5. ( bias correction ) v_{t} = v_{t-1} / (1 - beta_1 ^ t)
            6. ( bias correction ) s_{t} = s_{t-1} / (1 - beta_2 ^ t)
            7. w = w - lr * v_{t} / (sqrt(s_{t} + epsilon))
        """
        self.t += 1
        for p_name, params in self.model.parameters():
            g = params.grad.reshape(params.shape)
            g_2 = np.power(g, 2)

            if p_name not in self.v:
                self.v[p_name] = g
                self.s[p_name] = g_2
            else:
                self.v[p_name] = self.beta_1 * self.v[p_name] + (1 - self.beta_1) * g
                self.s[p_name] = self.beta_2 * self.s[p_name] + (1 - self.beta_2) * g_2

            if self.bias_fix:
                self.v[p_name] = self.v[p_name] / (1 - self.beta_1 ** self.t)
                self.s[p_name] = self.s[p_name] / (1 - self.beta_2 ** self.t)

            params.value = params.value - self.lr * self.v[p_name] / np.sqrt(self.s[p_name] + self.epsilon)


if __name__ == "__main__":
    x = dict()

    name = ['1', '2', '3', '4', '5', '6', '7']
    v = 0

    for i in name:
        x[i] = v
        v += 2

    print(x)
