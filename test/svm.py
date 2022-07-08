import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm

# raw_data = loadmat('svm_data/ex6data1.mat')
# data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
# data['y'] = raw_data['y']
#
# positive = data[data['y'].isin([1])]
# negative = data[data['y'].isin([0])]
#
# # plt.show()
#
# svms = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
# svms.fit(data[['X1', 'X2']], data['y'])
# print(svms.score(data[['X1', 'X2']], data['y']))
#
# w = svms.coef_[0]
# b = svms.intercept_[0]  # 训练后分类面的参数 w[0]*x[0] + w[1]*x[1] + b = 0
#
# x = np.linspace(0, 4, 50)  # 画分类面
# y = (-x * w[0] - b) / w[1]
#
#
#
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
# ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
# ax.legend()
# plt.plot(x, y)
# plt.show()




# raw_data = loadmat('svm_data/ex6data2.mat')
#
# data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
# data['y'] = raw_data['y']
#
# positive = data[data['y'].isin([1])]
# negative = data[data['y'].isin([0])]
#
# fig, ax = plt.subplots(figsize=(8,4))
# ax.scatter(positive['X1'], positive['X2'], s=30, marker='x', label='Positive')
# ax.scatter(negative['X1'], negative['X2'], s=30, marker='o', label='Negative')
# ax.legend()
# plt.show()
#
# def gaussian_kernel(x1, x2, sigma):
#     return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))
#
#
# non_svc = svm.SVC(C=100, gamma=10, probability=True)
# non_svc.fit(data[['X1', 'X2']], data['y'])
# print(non_svc.score(data[['X1', 'X2']], data['y']))
# data['Probability'] = non_svc.predict_proba(data[['X1', 'X2']])[:, 0]
#
# fig, ax = plt.subplots(figsize=(8,4))
# ax.scatter(data['X1'], data['X2'], s=30, c=data['Probability'], cmap='Blues')
# plt.show()


raw_data = loadmat('svm_data/ex6data3.mat')

X = raw_data['X']
Xval = raw_data['Xval']
y = raw_data['y'].ravel()
yval = raw_data['yval'].ravel()

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score = 0
best_param = {'C': None, 'gamma': None}

for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X, y)
        score = svc.score(Xval, yval)
        if score > best_score:
            best_score = score
            best_param['C'] = C
            best_param['gamma'] = gamma

print(best_score, best_param)

spam_train = loadmat('svm_data/spamTrain.mat')
spam_test = loadmat('svm_data/spamTest.mat')

X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()

svc = svm.SVC()
svc.fit(X, y)
print('training accuracy = {}'.format(svc.score(X, y)))
print('test accuracy = {}'.format(svc.score(Xtest, ytest)))
