import scipy as sp
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def plot_pca_scatter(x, y):
    px = x[:, 0]
    py = x[:, 1]
    pz = x[:, 2]

    y = np.choose(y.astype(np.int), [0, 2, 1])
    ax.scatter(px, py, pz, c=y.astype(np.float))

data = pd.read_csv("No_ones_or_zeroes.csv", index_col=False)

# X = data[1:, 1:36]
# y = data[1:, 0]
print(data.shape)

digit_1 = data.values[:, 1:16]
digit_1 = pd.DataFrame(digit_1)
print(type(digit_1))
digit_2 = data.values[:, 23:36]
digit_2 = pd.DataFrame(digit_2)
print(digit_2.shape)
digits = pd.concat([digit_1, digit_2], axis=1)
# for item in digit_2:
#     df2 = pd.DataFrame.from_records([digit_1])
# digits = pd.concat([digits, df2], ignore_index=True, axis=1)
print(digits.shape)
#digits = pd.concat([digit_1, digit_2], axis=1)
y_data = data.values[:, 17:22]
X = digits
y = y_data
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')


plot_pca_scatter(X, y)

# estimator = KMeans(n_clusters = 4)
# estimator.fit(X)
#
# fig = plt.figure(2)
# plt.clf()
#
# ax = fig.add_subplot(111, projection='3d')
#
# labels = estimator.labels_
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(np.float))

plt.show()

