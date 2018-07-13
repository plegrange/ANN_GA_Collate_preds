from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn import preprocessing

import scipy as sp

def dummyEncode(df):
    columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding ' + feature)
    return df

def plot_pca_scatter():
    colors = ['black', 'pink', 'purple', 'yellow', 'blue',
                'red', 'lime', 'cyan', 'orange', 'gray']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(colors)):

        px = data['Lift Height actual']#= X_pca[:, 0][y_data == i]
        py =data['Energy']#= X_pca[:, 1][y_data == i]
        pz =data['Weldcurrent actual']#= X_pca[:, 2][y_data == i]
        plt.title("3D Scatter plot for stud")
        plt.xlabel("Lift Height actual")
        plt.ylabel("Energy actual")
        ax.scatter(px, py, pz)


    # plt.legend(digits.target_names)
    plt.show()

def change_column_order(df, col_name, index):
    cols = df.columns.tolist()
    cols.remove(col_name)
    cols.insert(index, col_name)
    return df[cols]

data = pd.read_csv("No_ones_or_zeroes.csv", index_col=False)


# del_cols = ['Error']
# print()
# print("columns removed..........%s ",len(del_cols))
#
# for entry in del_cols:
#     del data[entry]
#
# data = change_column_order(data, 'Type', len(data.columns))
#
# for col in ['Type']:
#     data[col] = data[col].astype('category')
#
# data = dummyEncode(data)
#
# g = data.groupby('Type').count()
# counter = g.values[:, 0][1] - 1
#
# success = data[data['Type'] == 0][:counter-1]
# failure = data[data['Type'] == 1]
#
# data = pd.concat([success, failure], axis=0)
# data = shuffle(data)

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
y_data = data.values[:, 22]

print(data.head(5))

# scaleX = preprocessing.StandardScaler().fit(digits)
# #scaleX = preprocessing.MinMaxScaler().fit(X)
# digits = scaleX.transform(digits)

estimator = PCA(n_components=3)
X_pca = estimator.fit_transform(digits)

# X_pca = X_pca[X_pca[:,0] < 600]
# y_data = y_data[X_pca[:,0] < 600]
#
# X_pca = X_pca[X_pca[:,1] < 600]
# y_data = y_data[X_pca[:,1] < 600]
#
# X_pca = X_pca[X_pca[:,1] > -40]
# y_data = y_data[X_pca[:,1] > -40]

print(X_pca)

# X_pca = X_pca[X_pca[1] < 600]
# y_data = y_data[X_pca[1] < 600]

plot_pca_scatter()

