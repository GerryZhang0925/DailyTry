#########################################################################################
# Configuration
#########################################################################################
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

#########################################################################################
# Read Data
#########################################################################################
# ダウンロード元URL
url = 'https://archive.ics.uci.edu/ml/\
machine-learning-databases/00275/\
Bike-Sharing-Dataset.zip'

# 公開データのダウンロードと解凍
!wget $url -O Bike-Sharing-Dataset.zip | tail -n 1
!unzip -o Bike-Sharing-Dataset.zip | tail -n 1
!head -5 day.csv

# day.csvをデータフレームに取り込み
# 日付を表す列はparse_datesで指定する
df = pd.read_csv('day.csv', parse_dates=[1])

# データ属性の確認
print(df.dtypes)

# instant は連番で予測で不要なので削除
df = df.drop('instant', axis=1)

# 項目名の日本語化

columns = [
    '日付',  '季節',  '年', '月', '祝日', '曜日', '勤務日', '天気', 
    '気温', '体感温度',  '湿度', '風速',
    '臨時ユーザー利用数', '登録ユーザー利用数', '全体ユーザー利用数'
]

# 項目名を日本語に置き換え
df.columns = columns

#########################################################################################
# Confirm Data
#########################################################################################
# 先頭5行の確認
display(df.head())

# 最終5行の確認
display(df.tail())

# 度数分布表示

# グラフのサイズ調整のためのおまじない
from pylab import rcParams
rcParams['figure.figsize'] = (12, 12)

# データフレームの数値項目でヒストグラム表示
df.hist(bins=20)
plt.tight_layout()
plt.show()

# 欠損値チェック
df.isnull().sum()

#########################################################################################
# Time series display
#########################################################################################
# 時系列グラフの描画 (登録利用者数)
plt.figure(figsize=(12,4))

# グラフ描画
plt.plot(df['日付'],df['登録ユーザー利用数'],c='b')

# 方眼表示など
plt.grid()
plt.title('登録ユーザー利用数')

# 画面出力
plt.show()

#########################################################################################
# Preprocessing and Data spliting
#########################################################################################
# x, yへの分割
x = df.drop(['日付', '臨時ユーザー利用数', '登録ユーザー利用数',
    '全体ユーザー利用数'], axis=1)
y = df['登録ユーザー利用数'].values

# 分割日 mdayの設定
mday = pd.to_datetime('2012-11-1')

# 訓練用indexと検証用indexを作る
train_index = df['日付'] < mday
test_index = df['日付'] >= mday

# 入力データの分割
x_train = x[train_index]
x_test = x[test_index]

# yも同様に分割
y_train = y[train_index]
y_test = y[test_index]

# 日付データの分割(グラフ表示用)
dates_test = df['日付'][test_index]

# 結果確認(サイズを確認)
print(x_train.shape)
print(x_test.shape)

# 結果確認 (境界値を重点的に)
display(x_train.tail())
display(x_test.head())

# 目的変数の分割結果確認
print(y_train[:10])

#########################################################################################
# Apply algorithm
#########################################################################################
# アルゴリズム選定
# XGBRegressorを選定する
from xgboost import XGBRegressor
algorithm = XGBRegressor(objective ='reg:squarederror',
                         random_state=random_seed)

# 学習
algorithm.fit(x_train, y_train)

# 予測
y_pred = algorithm.predict(x_test)

# 予測結果確認
print(y_pred[:5])

# score関数の呼び出し
score = algorithm.score(x_test, y_test)

# R2値の計算
from sklearn.metrics import r2_score
r2_score = r2_score(y_test, y_pred)

# 結果確認
print(f'score: {score:.4f}  r2_ score: {r2_score:.4f}')

#正解データと予測結果を散布図で比較 (登録ユーザー利用数)
plt.figure(figsize=(6,6))
y_max = y_test.max()
plt.plot((0,y_max), (0, y_max), c='k')
plt.scatter(y_test, y_pred, c='b')
plt.title(f'正解データと予測結果の散布図(登録ユーザー利用数)\
  R2={score:.4f}')
plt.grid()
plt.show()

#########################################################################################
# Time series display
#########################################################################################
# 時系列グラフの描画 (登録ユーザー利用数)
import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(8, 4))

# グラフ描画
ax.plot(dates_test, y_test, label='正解データ', c='k')
ax.plot(dates_test, y_pred, label='予測結果', c='b')

# 日付目盛間隔
# 木曜日ごとに日付を表示
weeks = mdates.WeekdayLocator(byweekday=mdates.TH)
ax.xaxis.set_major_locator(weeks)

# 日付表記を90度回転
ax.tick_params(axis='x', rotation=90)

# 方眼表示など
ax.grid()
ax.legend()
ax.set_title('登録ユーザー利用数予測')

# 画面出力
plt.show()

#########################################################################################
# Tunning
#########################################################################################
# 項目をone hot encodeするための関数
def enc(df, column):
    df_dummy = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df.drop([column],axis=1),df_dummy],axis=1)
    return df

# 項目「月」「季節」をone hot encodingする

x2 = x.copy()
x2 = enc(x2, '月')
x2 = enc(x2, '季節')

# 結果確認
display(x2.head())



# 登録利用者モデル(チューニング後)

# mdayを基準に入力データを分割
x2_train = x2[train_index]
x2_test = x2[test_index]

#　アルゴリズム選定
algorithm2 = XGBRegressor(objective ='reg:squarederror',
    random_state=random_seed)

# 学習
algorithm2.fit(x2_train, y_train)

# 予測
y_pred2 = algorithm2.predict(x2_test)

# 予測結果確認
print(y_pred2[:5])

# 評価(登録利用者) (チューニング後)

# score関数の呼び出し
score2 = algorithm2.score(x2_test, y_test)

# 結果確認
print(f'score: {score2:.4f}')

#正解データと予測結果を散布図で比較 (登録利用者)
plt.figure(figsize=(6,6))
y_max = y_test.max()
plt.plot((0,y_max), (0, y_max), c='k')
plt.scatter(y_test, y_pred2, c='b')
plt.title(f'正解データと予測結果の散布図(登録ユーザー利用数) R2={score2:.4f}')
plt.grid()
plt.show()

# 時系列グラフの描画 (チューニング後 登録ユーザー利用数)
import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(8, 4))

# グラフ描画
ax.plot(dates_test, y_test, label='正解データ', c='k')
ax.plot(dates_test, y_pred, label='予測結果1', c='c')
ax.plot(dates_test, y_pred2, label='予測結果2', c='b')

# 日付目盛間隔
weeks = mdates.WeekdayLocator(byweekday=mdates.TH)
ax.xaxis.set_major_locator(weeks)

# 日付書式
ax.tick_params(axis='x', rotation=90)

# 方眼表示など
ax.grid()
ax.legend()
ax.set_title('登録ユーザー利用数予測')

# 画面出力
plt.show()

#########################################################################################
# Importance Analysis
#########################################################################################
# 登録ユーザー利用数に対する重要度分析
import xgboost as xgb
fig, ax = plt.subplots(figsize=(8, 4))
xgb.plot_importance(algorithm, ax=ax, height=0.8,
    importance_type='gain', show_values=False,
    title='重要度分析(登録ユーザー利用数)')
plt.show()
