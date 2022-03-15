import numpy as np
from metalearn import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt
models = linear_model.LinearRegression()


data_path = './linear_regression_data/ex1data1.txt'


def data_loader(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(',')
            line[0] = float(line[0])
            line[1] = float(line[1])
            dataset.append(line)

    return np.array(dataset)


dataset = data_loader(data_path)
x = [[1], [2], [3]]
model = LinearRegression(dataset, normalization=True)
model.train(100)
model.show()
model.get_model_info()
y = model.predict(x)
print(y)