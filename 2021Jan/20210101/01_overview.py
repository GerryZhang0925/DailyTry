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

#################################################################
# Read the data
#################################################################

# Prepare for data processing
from sklearn.datasets import load_breast_cancer

# Load Data
cancer = load_breast_cancer()

# Read the description of data
print(cancer.DESCR)

# Read data frame
columns = [
    '半径_平均', 'きめ_平均', '周長_平均', '面積_平均',
    '平滑度_平均', 'コンパクト度_平均', '凹面_平均',
    '凹点_平均', '対称性_平均', 'フラクタル度_平均',
    '半径_標準誤差', 'きめ_標準誤差', '周長_標準誤差',
    '面積_標準誤差', '平滑度_標準誤差',
    'コンパクト度_標準誤差', '凹面_標準誤差', '凹点_標準誤差',
    '対称性_標準誤差', 'フラクタル度_標準誤差',
    '半径_最大', 'きめ_最大', '周長_最大', '面積_最大',
    '平滑度_最大', 'コンパクト度_最大', '凹面_最大', '凹点_最大',
    '対称性_最大', 'フラクタル度_最大'
]

# Read data to the data frame
df = pd.DataFrame(cancer.data, columns=columns)

# Get results of the data set
y = pd.Series(cancer.target)

#################################################################
# Confirm the data
#################################################################
display(df[20:25])
print(y[120:25])

# confirm the sharp of the data
print(df.shape)
print()

# count the correct result
print(y.value_counts())

df0 = df[y==0]
df1 = df[y==1]

display(df0.head())
display(df1.head())

# configuration of figure size
plt.figure(figsize=(6, 6))

# Target variable == 0
plt.scatter(df0['半径_平均'], df0['きめ_平均'], marker='x', c='b', label='悪性')

# Target variable == 1
plt.scatter(df1['半径_平均'], df1['きめ_平均'], marker='s', c='k', label='良性')

# Display
plt.grid()

# Display labels
plt.xlabel('半径_平均')
plt.ylabel('きめ_平均')

plt.legend()
plt.show()

#################################################################
# Preprocessing
#################################################################
input_columns = ['半径_平均', 'きめ_平均']
x = df[input_columns]
display(x.head())

#################################################################
# Splitting
#################################################################
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state = random_seed)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#################################################################
# Algorithm Selection
#################################################################
from sklearn.linear_model import LogisticRegression
algorithm = LogisticRegression(random_state = random_seed)

#################################################################
# Learning
#################################################################
algorithm.fit(x_train, y_train)
print(algorithm)

#################################################################
# Prediction
#################################################################
y_pred = algorithm.predict(x_test)

print(y_pred)

#################################################################
# Evaluation
#################################################################
y_test10 = y_test[:10].values
print(y_test10)

y_pred10 = y_pred10[:10]
print(y_pred10)

w1 = (y_test10 == y_pred10)
print(w1)

w2 = w1.sum()
print(w2)

w = (y_test.values == y_pred)
correct = w.sum()

N = len(w)

score = correct / N

print(f'精度: {score:.04f}')

score = algorithm.score(x_test, y_test)
print(f'score: {score:.04f}')

#################################################################
# Tuning
#################################################################

x2_train, x2_test, y_train, y_test = train_set_split(df, y, train_size=0.7, test_size=0.3, random_state = random_seed)
algorithm2 = LogisticRegression(random_state_random_seed)

algorithm2.fit(x2_train, y_train)

score2 = algorithm2.score(x2_test, y_test)
print(f'Score:{score2:04f}')
