"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: bayes_demo.py
    Description: This demo tests gaussian bayes model.

    Created by Melrose-Lbt 2022-4-6
"""
import numpy as np
import matplotlib.pyplot as plt

male_path = './bayes_data/MALE.TXT'
female_path = './bayes_data/FEMALE.TXT'
test_path = './bayes_data/test1.txt'

P_w1 = 0.5
P_w2 = 0.5

judge_matrix = [[0, 7], [3, 0]]


def data_loader(path):
    dataset = []
    line_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\t', ' ')
            line = line.strip('\n').split(' ')
            for item in line:
                if item == '':
                    continue
                else:
                    line_list.append(float(item))
            dataset.append(line_list)
            line_list = []

    return np.array(dataset)


male = data_loader(male_path)
female = data_loader(female_path)
test = data_loader(test_path)

M1 = np.mean(male, axis=0)
M2 = np.mean(female, axis=0)

C1 = np.cov(male.T)
C2 = np.cov(female.T)

part_1 = np.log(P_w1) - 0.5 * np.log(np.linalg.det(C1))
part_2 = np.log(P_w2) - 0.5 * np.log(np.linalg.det(C2))

cnt = 0

# get real male and real female
real_male = test[test[:,-1] == 1][:, :-1]
real_female = test[test[:,-1] == 2][:, :-1]

pred_male = []
pred_male2 = []
pred_female = []
pred_female2 = []
for item in test:
    X = item[:-1]
    label = item[-1]
    d1 = part_1 - 0.5 * np.matmul(np.matmul((X - M1), np.linalg.inv(C1)), (X - M1).T)
    d2 = part_2 - 0.5 * np.matmul(np.matmul((X - M2), np.linalg.inv(C2)), (X - M2).T)

    d3 = judge_matrix[0][1] * d2
    d4 = judge_matrix[1][0] * d1

    y = d1 - d2
    if y > 0:
        y = 1
        pred_male.append(X)
    else:
        y = 2
        pred_female.append(X)

    if y == label:
        cnt += 1

    if d3 < d4:
        pred_male2.append(X)
    else:
        pred_female2.append(X)

pred_male = np.array(pred_male)
pred_female = np.array(pred_female)

pred_male2 = np.array(pred_male2)
pred_female2 = np.array(pred_female2)

plt.figure()
plt.scatter(real_male[:, 1], real_male[:, 0], color='b')
plt.scatter(real_female[:, 1], real_female[:, 0], color='r')
plt.title('Real data')
plt.show()

plt.figure()
plt.scatter(pred_male[:, 1], pred_male[:, 0], color='b')
plt.scatter(pred_female[:, 1], pred_female[:, 0], color='r')
plt.title('Predict data')
plt.show()

plt.figure()
plt.scatter(pred_male2[:, 1], pred_male2[:, 0], color='b')
plt.scatter(pred_female2[:, 1], pred_female2[:, 0], color='r')
plt.title('Predict data')
plt.show()