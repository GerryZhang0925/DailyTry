################################################################################
# Configuration
################################################################################
# 余分なワーニングを非表示にする
import warnings
warnings.filterwarnings('ignore')

# 必要ライブラリのimport
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# matplotlib日本語化対応
import japanize_matplotlib

# データフレーム表示用関数
from IPython.display import display

# 表示オプション調整
# numpyの浮動小数点の表示精度
np.set_printoptions(suppress=True, precision=4)

# pandasでの浮動小数点の表示精度
pd.options.display.float_format = '{:.4f}'.format

# データフレームですべての項目を表示
pd.set_option("display.max_columns",None)

# グラフのデフォルトフォント指定
plt.rcParams["font.size"] = 14

# 乱数の種
random_seed = 123

################################################################################
# Read Data
################################################################################
# ダウンロード元URL
url = 'https://archive.ics.uci.edu/ml/\
machine-learning-databases/00275/\
Bike-Sharing-Dataset.zip'

# 公開データのダウンロードと解凍
!wget $url -O Bike-Sharing-Dataset.zip | tail -n 1
!unzip -o Bike-Sharing-Dataset.zip | tail -n 1

# day.csvをデータフレームに取り込み
# 日付を表す列はparse_datesで指定する
df = pd.read_csv('day.csv', parse_dates=[1])

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

# 先頭5行の確認
display(df.head())

# 最終5行の確認
display(df.tail())

################################################################################
# Preprocessing & Data Splitting
################################################################################
# 「日付」と「登録ユーザー利用数」のみ抽出し、
# 列名を日付:ds 、登録ユーザー利用数:y に置き換えたデータフレームdf2を作る

# データフレーム全体のコピー
df2 = df.copy()

# 「日付」「登録ユーザー利用数」列の抽出
df2 = df2[['日付', '登録ユーザー利用数']]

# 列名の置き換え
df2.columns = ['ds', 'y']

# 結果確認
display(df2.head())

# 分割日 mdayの設定
mday = pd.to_datetime('2012-11-1')

# 訓練用indexと検証用indexを作る
train_index = df2['ds'] < mday
test_index = df2['ds'] >= mday

# 入力データの分割
x_train = df2[train_index]
x_test = df2[test_index]

# 日付データの分割(グラフ表示用)
dates_test = df2['ds'][test_index]

################################################################################
# Apply Algorithm
################################################################################
# ライブラリのimport
from fbprophet import Prophet

# モデル選定
# 3つのseasonalityパラメータの設定が重要 
# 今回のデータの場合、日単位のデータなのでdaily_seasonalityは不要
# weekly_seasonality とdaily_seasonalityは 
# True / Falseの他に数値で指定することも可能 (三角関数の個数)
# seasonality_mode: additive(デフォルト) multiplicative

m1 = Prophet(yearly_seasonality=True, weekly_seasonality=True, 
    daily_seasonality=False,
    seasonality_mode='multiplicative')

# 学習
m1.fit(x_train)

# 予測用データの作成
# (日付 ds だけの入ったデータフレーム)
# 61は予測したい日数 (2012-11-1 から2012-12-31)
future1 = m1.make_future_dataframe(periods=61, freq='D')

# 結果確認
display(future1.head())
display(future1.tail())

# 予測
# 結果はデータフレームで戻ってくる
fcst1 = m1.predict(future1)

# 要素ごとのグラフ描画
# この段階ではトレンド、週周期、年周期
fig = m1.plot_components(fcst1)
plt.show()

################################################################################
# Visulization
################################################################################
# 訓練データ・検証データ全体のグラフ化
fig, ax = plt.subplots(figsize=(10,6))

# 予測結果のグラフ表示(prophetの関数)
m1.plot(fcst1, ax=ax)

# タイトル設定など
ax.set_title('登録ユーザー利用数予測')
ax.set_xlabel('日付')
ax.set_ylabel('利用数')

# グラフ表示
plt.show()



# ypred1: fcst1から予測部分のみ抽出する
ypred1 = fcst1[-61:][['yhat']].values

# ytest1: 予測期間中の正解データ
ytest1 = x_test['y'].values

# R2値の計算
from sklearn.metrics import r2_score
score = r2_score(ytest1, ypred1)

# 結果確認
print(f'R2 score:{score:.4f}')

# 時系列グラフの描画 
import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(8, 4))

# グラフ描画
ax.plot(dates_test, ytest1, label='正解データ', c='k')
ax.plot(dates_test, ypred1, label='予測結果', c='b')

# 日付目盛間隔
# 木曜日ごとに日付を表示
weeks = mdates.WeekdayLocator(byweekday=mdates.TH)
ax.xaxis.set_major_locator(weeks)

# 日付表記を90度回転
ax.tick_params(axis='x', rotation=90)

# 方眼表示など
ax.grid()
ax.legend()
ax.set_title('登録ユーザー利用数予測結果')

# 画面出力
plt.show()

################################################################################
# Tunning 1
################################################################################
# ステップ1 「休日」を特別な日として追加
# ステップ2 回帰モデルに「天気」「気温」「風速」「湿度」を追加
# 休日の抽出
df_holiday = df[df['祝日']==1]
holidays = df_holiday['日付'].values

# データフレーム形式に変換
df_add = pd.DataFrame({'holiday': 'holi',
    'ds': holidays,
    'lower_window': 0,
    'upper_window': 0
})

# 結果確認
display(df_add.head())     
display(df_add.tail())

# 休日(df_add)をモデルの入力とする

# アルゴリズム選定
# holidaysパラメータを追加してモデルm2を生成
m2 = Prophet(yearly_seasonality=True, 
    weekly_seasonality=True, daily_seasonality=False,
    holidays = df_add, seasonality_mode='multiplicative')

# 学習
m2 = m2.fit(x_train)

# 予測
fcst2 = m2.predict(future1)

# 要素ごとのグラフ描画
fig = m2.plot_components(fcst2)
plt.show()



# R値の計算

# fcst2から予測部分のみ抽出する
ypred2 = fcst2[-61:][['yhat']].values

# R2値の計算
score2 = r2_score(ytest1, ypred2)

# 結果確認
r2_text2 = f'R2 score:{score2:.4f}'
print(r2_text2)



# 時系列グラフの描画 
import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(8, 4))

# グラフ描画
ax.plot(dates_test, ytest1, label='正解データ', c='k')
ax.plot(dates_test, ypred1, label='予測結果v1', c='c')
ax.plot(dates_test, ypred2, label='予測結果v2', c='b')

# 日付目盛間隔
# 木曜日ごとに日付を表示
weeks = mdates.WeekdayLocator(byweekday=mdates.TH)
ax.xaxis.set_major_locator(weeks)

# 日付表記を90度回転
ax.tick_params(axis='x', rotation=90)

# 開始日と終了日
sday = pd.to_datetime('2012-11-1')
eday = pd.to_datetime('2013-1-1')
ax.set_xlim(sday, eday) 

# 方眼表示など
ax.grid()
ax.legend()
ax.set_title('登録ユーザー利用数予測結果  ' + r2_text2)

# 画面出力
plt.show()

################################################################################
# Tunning 2
################################################################################
# 学習データに「天気」「気温」「風速」「湿度」を追加
df3 = pd.concat([df2, df[['天気', '気温', '風速', '湿度']]], axis=1)

# 入力データの分割
x2_train = df3[train_index]
x2_test = df3[test_index]

# 結果確認
display(x2_train.tail())

# アルゴリズム選定

m3 = Prophet(yearly_seasonality=True, 
    weekly_seasonality=True, daily_seasonality=False,
    seasonality_mode='multiplicative', holidays = df_add)

#  add_regressor関数で、「天気」「気温」「風速」「湿度」をモデルに組み込む
m3.add_regressor('天気')
m3.add_regressor('気温')
m3.add_regressor('風速')
m3.add_regressor('湿度')

# 学習
m3.fit(x2_train)

# 予測用の入力データを作る
future3 = df3[['ds', '天気', '気温', '風速', '湿度']]

# 予測
fcst3 = m3.predict(future3)

# 要素ごとのグラフ描画
fig = m3.plot_components(fcst3)
plt.show()



# R値の計算

# fcstから予測部分のみ抽出する
ypred3 = fcst3[-61:][['yhat']].values
score3 = r2_score(ytest1, ypred3)

# 結果確認
r2_text3 = f'R2 score:{score3:.4f}'
print(r2_text3)

# 時系列グラフの描画 
import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(8, 4))

# グラフ描画
ax.plot(dates_test, ytest1, label='正解データ', c='k')
ax.plot(dates_test, ypred2, label='予測結果v2', c='c')
ax.plot(dates_test, ypred3, label='予測結果v3', c='b')

# 日付目盛間隔
# 木曜日ごとに日付を表示
weeks = mdates.WeekdayLocator(byweekday=mdates.TH)
ax.xaxis.set_major_locator(weeks)

# 日付表記を90度回転
ax.tick_params(axis='x', rotation=90)

# 方眼表示など
ax.grid()
ax.legend()
ax.set_title('登録ユーザー利用数予測結果  ' + r2_text3)

# 画面出力
plt.show()

################################################################################
# アイスクリーム購買予測」で時系列分析
# https://www.icecream.or.jp/biz/data/expenditures.html
################################################################################
# データ読み込み
url2 = 'https://github.com/makaishi2/\
sample-data/blob/master/data/ice-sales.xlsx?raw=true'

df = pd.read_excel(url2, sheet_name=0)

# データ確認
display(df.head())
display(df.tail())

# 時系列グラフの描画 (アイスクリーム支出金額)
fig, ax = plt.subplots(figsize=(12, 4))

# グラフ描画
ax.plot(df['年月'], df['支出'],c='b')

# 3か月区切りの目盛にする
month3 = mdates.MonthLocator(interval=3)
ax.xaxis.set_major_locator(month3)

# 日付表記を90度回転
ax.tick_params(axis='x', rotation=90)

# 開始日と終了日
sday = pd.to_datetime('2015-1-1')
eday = pd.to_datetime('2019-12-31')
ax.set_xlim(sday, eday) 

# 方眼表示など
ax.grid()
ax.set_title('アイスクリーム支出金額')

# 画面出力
plt.show()

# データ前処理
# データ形式をProphet用に合わせる
x = df.copy()
x.columns = ['ds', 'y']
display(x.head())

# データ分割
# 2019年1月を基準に訓練データと検証データを分割
# 分割日 mdayの設定
mday = pd.to_datetime('2019-1-1')

# 訓練用indexと検証用indexを作る
train_index = x['ds'] < mday
test_index = x['ds'] >= mday

# 入力データの分割
x_train = x[train_index]
x_test = x[test_index]

#日付列もグラフ描画のために分割
dates_train = x['ds'][train_index]
dates_test = x['ds'][test_index]

# アルゴリズムの選択
# ライブラリのimport
from fbprophet import Prophet
m = Prophet(yearly_seasonality=5, weekly_seasonality=False, daily_seasonality=False)

# 学習
m = m.fit(x_train)

# 予測
future = x[['ds']]
fcst = m.predict(future)

# 評価

# fcstから予測部分のみ抽出する
ypred = fcst[-12:]['yhat'].values

# 正解データのリスト
ytest = x_test['y'].values

# R値の計算
from sklearn.metrics import r2_score
score = r2_score(ytest, ypred)
score_text = f'R2 score:{score:.4f}'
print(score_text)

# 時系列グラフの描画 (アイスクリーム支出金額)
fig, ax = plt.subplots(figsize=(8, 4))

# グラフ描画
ax.plot(dates_test, ytest, label='正解データ', c='k')
ax.plot(dates_test, ypred, label='予測結果', c='b')

# 1か月区切りの目盛にする
month = mdates.MonthLocator()
ax.xaxis.set_major_locator(month)

# 日付表記を90度回転
ax.tick_params(axis='x', rotation=90)

# 開始日と終了日
sday = pd.to_datetime('2019-1-1')
eday = pd.to_datetime('2019-12-1')
ax.set_xlim(sday, eday) 

# 方眼表示など
ax.grid()
ax.legend()
ax.set_title('アイスクリーム支出金額予測　' + score_text)

# 画面出力
plt.show()
