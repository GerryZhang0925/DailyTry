###################################################################################
# Configuration
###################################################################################
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

###################################################################################
# Algorithm Selection
###################################################################################
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

test_size = 0.1

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=test_size, random_state=random_seed,
                                                    stratify=y)
print(x.shape)
print(x_train.shape)
print(x_test.shape)

# Linear Regression
from sklearn.linear_model import LogisticRegression
algorithm1 = LogisticRegression(random_state=random_seed)

# SVM
from sklearn.svm import SVC
algorithm2 = SVC(kernel='rbf', random_state=random_seed)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
algorithm3 = DecisionTreeClassifier(random_state=random_seed)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
algorithm4 = RandomForestClassifier(random_state=random_seed)

# XGBoost
from xgboost import XGBClassifier
algorithm5 = XGBClassifier(random_state=random_seed)

# Make a list of algorithm
algorithms = [algorithm1, algorithm2, algorithm3, algorithm4, algorithm5]

for algorithm in algorithms:
    algorithm.fit(x_train, y_train)
    score = algorithm.score(x_test, y_test)
    name = algorithm.__class__.__name__
    print(f'score: {score:.4f} {name}')

###################################################################################
# Optimization of hyper parameters
###################################################################################
algorithm = SVC(kernel='rbf', random_state=random_seed)
print(algorithm)

# Find gamma value
gammas = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

for gamma in gammas:
    algorithm = SVC(kernel='rbf', gamma=gamma, random_state=random_seed)
    algorithm.fit(x_train, y_train)
    score = algorithm.score(x_test, y_test)
    print(f'score: {score:.4f} gamma: {gamma}')

# Find C value
Cs = [1, 10, 100, 1000, 10000]
for C in Cs:
    algorithm = SVC(kernel='rbf', gamma=0.001, C=C, random_state=random_seed)
    algorithm.fit(x_train, y_train)
    score = algorithm.score(x_test, y_test)
    print(f'score: {score:.4f} C: {C}')


###################################################################################
# Cross Validation
###################################################################################
algorithm = SVC(kernel='rbf', random_state=random_seed, gamma=0.001, C=1)

from sklearn.model_selection import StratifiedKFold
stratifiedkfold = StratifiedKFold(n_splits=3)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(algorithm, x_train, y_train, cv=stratifiedkfold)

mean = scores.mean()

print(f'平均スコア:{mean:.4f} 個別スコア:{scores}')

# Use cross validation to select algorithm
from sklearn.linear_model import LogisticRegression
algorithm1 = LogisticRegression(random_state=random_seed)

from sklearn.svm import SVC
algorithm2 = SVC(kernel='rbf', random_state=random_seed, gamma=0.001, C=1)

from sklearn.tree import DecisionTreeClassifier
algorithm3 = DecisionTreeClassifier(random_state=random_seed)

from sklearn.ensemble import RandomForestClassifier
algorithm4 = RandomForestClassifier(random_state=random_seed)

from xgboost import XBGClassifier
algorithm5 = XGBClassifier(random_state=random_seed)
algorithms = [algorithm1, algorithm2, algorithm3, algorithm4, algorithm5]

from sklearn.model_selection import StratifiedKFold
stratifiedkfold = StratifiedKFold(n_splits=3)

from sklearn.model_selection import cross_val_score
for algorithm in algorithms:
    scores = cross_val_score(algorithm, x_train, y_train,
                             cv=stratifiedkfold)
    score = scores.mean()
    name = algorithm.__class__.__name__
    print(f'平均スコア:{score:.4f} 個別スコア:{scores} {name}')

###################################################################################
# Grid Search
###################################################################################
params = {
       'C':[1, 10, 100, 1000, 10000],
       'gamma':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
}
algorithm = SVC(random_state=random_seed)

from sklearn.model_selection import StratifiedkFold
stratifiedkfold = StratifiedkFold(n_splits=3)

from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(algorithm, params, cv=stratifiedkfold)
gs.fit(x_train, y_train)

best = gs.best_estimator_
best_pred = best.predict(x_test)
print(best)

score = best.score(x_test, y_test)
print(f'スコア: {score:.4f}')

from sklearn.metrics import confusion_matrix
print()
print('混同行列')
print(confusion_matrix(y_test, best_pred))
