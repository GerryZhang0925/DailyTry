######################################################
# Configuration
######################################################
# Do not show warnings
import warnings
warnings.filterwarnings('ignore')

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# matplotlib for japanese support
import japanize_matplotlib

# functions for data frame display
from IPython.display import display

# Adjust display options
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
pd.set_option("display.max_columns", None)
plt.rcParams["font.size"]=14
random_seed = 123

######################################################
# Data Creation
######################################################
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification

X1, y1 = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=random_seed, n_clusters_per_class=1, n_samples=200, n_classes=2)
X2, y2 = make_moons(noise=0.05, random_state=random_seed, n_samples=200)
X3, y3 = make_circles(noise=0.02, random_state=random_seed, n_samples=200)

DataList = [(X1, y1), (X2, y2), (X3, y3)]
N = len(DataList)


plt.figure(figsize=(15,4))

from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#0000FF', '#000000'])

for i, data in enumerate(DataList):
    X, y = data
    ax = plt.subplot(1, N, i+1)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap)

plt.show()

######################################################
# Logistic Regression
######################################################
def sigmoid(x):
    return 1/(1 + np.exp(-x))

x = np.linspace(-5, 5, 101)
y = sigmoid(x)
plt.plot(x, y, label='シグモイド関数', c='b', lw=2)
plt.legend()
plt.grid()
plt.show()

from sklearn.linear_model import LogisticRegression
algorithm = LogisticRegression(random_state=random_seed)

print(algorithm)


from sklearn.model_selection import train_test_split

# 決定境界の表示関数
def plot_boundary(ax, x, y, algorithm):
    x_train, x_test, y_train, y_test = train_test_split(x, y,
            test_size=0.5, random_state=random_seed)
    # カラーマップ定義
    from matplotlib.colors import ListedColormap
    cmap1 = plt.cm.bwr
    cmap2 = ListedColormap(['#0000FF', '#000000'])

    h = 0.005
    algorithm.fit(x_train, y_train)
    score_test = algorithm.score(x_test, y_test)
    score_train = algorithm.score(x_train, y_train)
    f1_min = x[:, 0].min() - 0.5
    f1_max = x[:, 0].max() + 0.5
    f2_min = x[:, 1].min() - 0.5
    f2_max = x[:, 1].max() + 0.5
    f1, f2 = np.meshgrid(np.arange(f1_min, f1_max, h), 
                         np.arange(f2_min, f2_max, h))
    if hasattr(algorithm, "decision_function"):
        Z = algorithm.decision_function(np.c_[f1.ravel(), f2.ravel()])
        Z = Z.reshape(f1.shape)
        ax.contour(f1, f2, Z, levels=[0], linewidth=2)
    else:
        Z = algorithm.predict_proba(np.c_[f1.ravel(), f2.ravel()])[:, 1]
        Z = Z.reshape(f1.shape)
    ax.contourf(f1, f2, Z, cmap=cmap1, alpha=0.3)
    ax.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap=cmap2)
    ax.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap=cmap2, marker='x')
    text = f'検証:{score_test:.2f}  訓練: {score_train:.2f}'
    ax.text(f1.max() - 0.3, f2.min() + 0.3, text, horizontalalignment='right',
    fontsize=18)

def plot_boundaries(algorithm, DataList):
    plt.figure(figsize=(15,4))
    for i, data in enumerate(DataList):
        X, y = data
        ax = plt.subplot(1, N, i+1)
        plot_boundary(ax, X, y, algorithm)
    plt.show()

plot_boundaries(algorithm, DataList)

######################################################
# SVM
######################################################
from sklearn.svm import SVC
algorithm = SVC(kernel='rbf')

print(algorithm)

plot_boundaries(algorithm, DataList)

######################################################
# Deep Learning (single layer)
######################################################
from sklearn.neural_network import MLPClassifier
algorithm = MLPClassifier(random_state=random_seed)

print(algorithm)

plot_boundaries(algorithm, DataList)

######################################################
# Deep Learning (double layer)
######################################################
from sklearn.neural_network import MLPClassifier
algorithm = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=random_seed)

print(algorithm)

plot_boundaries(algorithm, DataList)

######################################################
# Decision Tree
######################################################
from sklearn.tree import DecisionTreeClassifier
algorithm = DecisionTreeClassifier(random_state=random_seed)

print(algorithm)

plot_boundaries(algorithm, DataList)

from sklearn.tree import DecisionTreeClassifier
algorithm = DecisionTreeClassifier(max_depth=3,
                                   random_state=random_seed)

print(algorithm)

plot_boundaries(algorithm, DataList)

import seaborn as sns
df_iris = sns.load_dataset("iris")
df2 = df_iris[50:150]

X=df2.drop('species', axis=1)
y=df2['species']

from sklearn.tree import DecisionTreeClassifier
algorithm = DecisionTreeClassifier(random_state=random_seed)
algorithm.fit(X, y)

from sklearn import tree
with open('iris-dtree.dot', mode='w') as f:
    tree.export_graphviz(algorithm, out_file=f, 
                         feature_names=X.columns, filled=True, rounded=True,
                         special_characters=True, impurity=False, proportion=False)
import pydotplus
from IPython.display import Image
graph = pydotplus.graphviz.graph_from_dot_file('iris-dtree.dot')
graph.write_png('iris-dtree.png')
Image(graph.create_png())

######################################################
# Random Forest
######################################################
from sklearn.ensemble import RandomForestClassifier
algorithm = RandomForestClassifier(random_state=random_seed)
print(algorithm)
plot_boundaries(algorithm, DataList)

######################################################
# XGBoost
######################################################
import xgboost
algorithm = xgboost.XGBClassifier(random_state=random_seed)
print(algorithm)
plot_boundaries(algorithm, DataList)
