#################################################################################################
# Configuration
#################################################################################################
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

#################################################################################################
# Read Data
#################################################################################################
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip -0 bank.zip | tail -n 1
!unzip -o bank.zip | tail -n 1

df_all = pd.read_csv('bank-full.csv', sep=';')

columns = [
    '年齢', '職業', '婚姻', '学歴', '債務不履行', '平均残高',
    '住宅ローン', '個人ローン', '連絡手段', '最終通話日',
    '最終通話月', '最終通話秒数', '通話回数_販促中',
    '前回販促後_経過日数', '通話回数_販促前', '前回販促結果',
    '今回販促結果'
]
df_all.columns = columns

display(df_all.head())
print(df_all.shape)
print()

print(df_all['今回販促結果'].value_counts())
rate = df_all['今回販促結果'].value_counts()['yes']/len(df_all)
print(f'営業成功率: {rate:.4f}')
print(df_all.isnull().sum())

#################################################################################################
# Preprocessing
#################################################################################################
def enc(df, column):
    df_dummy = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df.drop([column], axis=1), df_dummy],axis=1)
    return df

df_all2 = df_all.copy()
df_all2 = enc(df_all2, '職業')
df_all2 = enc(df_all2, '婚姻')
df_all2 = enc(df_all2, '学歴')
df_all2 = enc(df_all2, '連絡手段')
df_all2 = enc(df_all2, '前回販促結果')

display(df_all2.head())

def enc_bin(df, column):
    df[column] = df[column].map(dict(yes=1, no=0))
    return df

df_all2 = enc_bin(df_all2, '債務不履行')
df_all2 = enc_bin(df_all2, '住宅ローン')
df_all2 = enc_bin(df_all2, '個人ローン')
df_all2 = enc_bin(df_all2, '今回販促結果')

display(df_all2.head())

month_dict = dict(jan=1, feb=2, mar=3, apr=4,
                  may=5, jun=6, jul=7, aug=8,
                  sep=9, oct=10, nov=11, dec=12)

def enc_month(df, column):
    df[column] = df[column].map(month_dict)
    return df

df_all2 = enc_month(df_all2, '最終通話月')
display(df_all2.head())

#################################################################################################
# Data Splitting & algorithm selection
#################################################################################################
x = df_all2.drop('今回販促結果', axis=1)
y = df_all2['今回販促結果'].values

test_size = 0.4

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, random_state=random_seed,
    stratify=y)

from sklearn.linear_model import LogisticRegression
algorithm1 = LogisticRegression(random_state=random_seed)

from sklearn.tree import DecisionTreeClassifier
algorithm2 = DecisionTreeClassifier(random_state=random_seed)

from sklearn.ensemble import RandomForestClassifier
algorithm3 = RandomForestClassifier(random_state=random_seed)

from xgboost import XGBClassifier
algorithm4 = XGBClassifier(random_state=random_seed)

algorithms = [algorithm1, algorithm2, algorithm3, algorithm4]

from sklearn.model_selection import StratifiedKFold
stratifiedkfold = StratifiedKfold(n_splits=3)

from sklearn.model_selection import cross_val_score
for algorithm in algorithms:
    scores = cross_val_score(algorithm, x_train, y_train,
                             cv=stratifiedkfold, scoring='roc_auc')
    score = scores.mean()
    name = algorithm.__class__.__name__
    print(f'平均スコア: {score:.4f}  個別スコア: {scores}  {name}')

#################################################################################################
# Valuation
#################################################################################################
def make_cm(matrix, columns):
    n = len(columns)
    act = ['正解データ'] * n
    pred = ['予測結果'] * n

    cm = pd.DataFrame(matrix, columns=[pred, columns], index=[act, columns])
    return cm

# アルゴリズム選定
# XGBoostを利用
algorithm = XGBClassifier(random_state=random_seed)

# 学習
algorithm.fit(x_train, y_train)

# 予測
y_pred = algorithm.predict(x_test)

# 評価
# 混同行列を出力
from sklearn.metrics import confusion_matrix
df_matrix = make_cm(
    confusion_matrix(y_test, y_pred), ['失敗', '成功'])
display(df_matrix)

# 適合率, 再現率, F値を計算
from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, _ = precision_recall_fscore_support(
    y_test, y_pred, average='binary')
print(f'適合率: {precision:.4f}  再現率: {recall:.4f}  F値: {fscore:.4f}')

#################################################################################################
# Tunning
#################################################################################################
# 確率値の度数分布グラフ
import seaborn as sns

# y=0の確率値取得
y_proba0 = algorithm.predict_proba(x_test)[:,1]

# y_test=0 と y_test=1 でデータ分割
y0 = y_proba0[y_test==0]
y1 = y_proba0[y_test==1]

# 散布図描画
plt.figure(figsize=(6,6))
plt.title('確率値の度数分布')
sns.distplot(y1, kde=False, norm_hist=True,
    bins=50, color='b', label='成功')
sns.distplot(y0, kde=False, norm_hist=True,
    bins=50, color='k', label='失敗')
plt.xlabel('確率値')
plt.legend()
plt.show()


# 閾値を変更した場合の予測関数の定義
def pred(algorithm, x, thres):
    # 確率値の取得(行列)
    y_proba = algorithm.predict_proba(x)
    
    # 予測結果1の確率値
    y_proba1 =  y_proba[:,1]
    
    # 予測結果1の確率値 > 閾値
    y_pred = (y_proba1 > thres).astype(int)
    return y_pred

# 閾値を0.05刻みに変化させて、適合率, 再現率, F値を計算する
thres_list = np.arange(0.5, 0, -0.05)

for thres in thres_list:
    y_pred = pred(algorithm, x_test, thres)
    pred_sum =  y_pred.sum()
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary')
    print(f'閾値: {thres:.2f} 陽性予測数: {pred_sum}\
 適合率: {precision:.4f} 再現率: {recall:.4f}  F値: {fscore:.4f})')


# F値を最大にする閾値は0.30
y_final = pred(algorithm, x_test, 0.30)

# 混同行列を出力
df_matrix2 = make_cm(
    confusion_matrix(y_test, y_final), ['失敗', '成功'])
display(df_matrix2)

# 適合率, 再現率, F値を計算
precision, recall, fscore, _ = precision_recall_fscore_support(
    y_test, y_final, average='binary')
print(f'適合率: {precision:.4f}  再現率: {recall:.4f}\
  F値: {fscore:.4f}')

#################################################################################################
# Importance Analysis
#################################################################################################
# 重要度ベクトルの取得
importances = algorithm.feature_importances_

# 項目名をキーにSeriesを生成
w = pd.Series(importances, index=x.columns)

# 値の大きい順にソート
u = w.sort_values(ascending=False)

# top10のみ抽出
v = u[:10]

# 重要度の棒グラフ表示
plt.title('入力項目の重要度')
plt.bar(range(len(v)), v, color='b', align='center')
plt.xticks(range(len(v)), v.index, rotation=90)
plt.show()

column = '前回販促結果_success'

sns.distplot(x_test[y_test==1][column], kde=False, norm_hist=True,
            bins=5,color='b', label='成功')
sns.distplot(x_test[y_test==0][column], kde=False, norm_hist=True,
             bins=5,color='k', label='失敗')

plt.legend()
plt.show()

column = '最終通話秒数'

sns.distplot(x_test[y_test==1][column], kde=False, norm_hist=True,
             bins=50, color='b', label='成功')
sns.distplot(x_test[y_test==0][column], kde=False, norm_hist=True,
             bins=50, color='k', label='失敗')

plt.legend()
plt.show()

column = '連絡手段_unknown'

sns.distplot(x_test[y_test==1][column], kde=False, norm_hist=True,
            bins=5,color='b', label='成功')
sns.distplot(x_test[y_test==0][column], kde=False, norm_hist=True,
             bins=5,color='k', label='失敗')

plt.legend()
plt.show()

column = '住宅ローン'

sns.distplot(x_test[y_test==1][column], kde=False, norm_hist=True,
            bins=5,color='b', label='成功')
sns.distplot(x_test[y_test==0][column], kde=False, norm_hist=True,
             bins=5,color='k', label='失敗')

plt.legend()
plt.show()

column = '婚姻_single'

sns.distplot(x_test[y_test==1][column], kde=False, norm_hist=True,
            bins=5,color='b', label='成功')
sns.distplot(x_test[y_test==0][column], kde=False, norm_hist=True,
             bins=5,color='k', label='失敗')

plt.legend()
plt.show()

