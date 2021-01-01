########################################################################
# Configuration
########################################################################
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

########################################################################
# Read data
########################################################################
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
x = cancer.data
y = 1 - cancer.target
x2 = x[:,:2]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.7, test_size=0.3, random_state=random_seed)

########################################################################
# Algorithm Selection and Evaluation
########################################################################
from sklearn.linear_model import LogisticRegression
algorithm = LogisticRegression(random_state=random_seed)

algorithm.fit(x_train, y_train)
y_pred = algorithm.predict(x_test)
score = algorithm.score(x_test, y_test)
print(f'score: {score:.4f}')

########################################################################
# Confusion Matrix
########################################################################
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, y_pred)
print(matrix)

def make_cm(matrix, columns):
    n = len(columns)
    act = ['正解データ'] * n
    pred = ['予測結果'] * n

    cm = pd.DataFrame(matrix, columns=[pred, columns], index=[act, columns])
    return cm

cm = make_cm(matrix, ['良性', '悪性'])
display(cm)

########################################################################
# Accuracy/Precision/Recall/F Value
########################################################################
from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, _= precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f'適合率: {precision:.4f}')
print(f'再現率: {recall: .4f}')
print(f'F値: {fscore: .4f}')

########################################################################
# Probability/Threshold
########################################################################
y_proba = algorithm.predict_proba(x_test)
print(y_proba[:10,:])

y_probal=y_proba[:,1]
print(y_test[10:20])
print(y_pred[10:20])
print(y_probal[10:20])

# Change the threshold
thres = 0.5
print((y_probal[10:20] > thres).astype(int))
thres = 0.7
print((y_probal[10:20] > thres).astype(int))

def pred(algorithm, x, thres):
    y_proba = algorithm.predict_proba(x)
    y_probal = y_proba[:, 1]
    y_pred = (y_probal > thres).astype(int)
    return y_pred

pred_05 = pred(algorithm, x_test, 0.5)
pred_07 = pred(algorithm, x_test, 0.7)

print(pred_05[10:20])
print(pred_07[10:20])

########################################################################
# Precision-Recall Curve
########################################################################
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_probal)
df_pr = pd.DataFrame([thresholds, precision, recall]).T
df_pr.columns = ['しきい値', '適合率', '再現率']
display(df_pr[52:122:10])

plt.figure(figsize=(6,6))
plt.fill_between(recall, precision, 0)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('再現率')
plt.ylabel('適合率')
plt.title('PR曲線')
plt.show()

from sklearn.metrics import auc
pr_auc = auc(recall, precision)
print(f'PR曲線下面積: {pr_auc:.4f}')

########################################################################
# ROC Curve
########################################################################
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_probal, drop_intermediate=False)

df_roc = pd.DataFrame([thresholds, fpr, tpr]).T
df_roc.columns = ['閾値', '偽陽性率', '敏感度']

display(df_roc[21:91:10])

plt.figure(figsize=(6,6))
plt.plot([0, 1], [0, 1], 'k--')
plt.fill_between(fpr, tpr, 0)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.xlabel('偽陽性率')
plt.ylabel('敏感度')
plt.title('ROC曲線')
plt.show()

roc_auc = auc(fpr, tpr)
print(f'ROC曲線下面積:{roc_auc:.4f}')

########################################################################
# Importance of parameters
########################################################################
import seaborn as sns
df_iris = sns.load_dataset("iris")
columns_i = ['萼片長', '萼片幅', '花弁長', '花弁幅', '種別']
df_iris.columns = columns_i
x = df_iris[['萼片長', '萼片幅', '花弁長', '花弁幅']]
y = df_iris['種別']

from sklearn.ensemble import RandomForestClassifier
algorithm = RandomForestClassifier(random_state=random_seed)

algorithm.fit(x, y)

importances = algorithm.feature_importance_
w = pd.Series(importances, index=x.columns)
u = w.sort_values(ascending=False)
print(u)

plt.bar(range(len(u)), u, color='b', align='center')
plt.xticks(range(len(u)), u.index, rotation=90)
plt.title('入力変数の重要度')
plt.show()
