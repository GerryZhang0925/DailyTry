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
# Read Data
######################################################
import seaborn as sns

df_titanic = sns.load_dataset("titanic")

columns_t = ['生存', '等室', '性別', '年齢', '兄弟配偶者数',
             '両親子供数', '料金', '乗船港コード', '等室名',
             '男女子供', '成人男子', 'デッキ', '乗船港', '生存可否', '独身']
df_titanic.columns = columns_t

######################################################
# Remove useless items
######################################################
df1 = df_titanic.drop('等室名', axis=1)
df2 = df1.drop('乗船港', axis=1)
df3 = df2.drop('生存可否', axis=1)
display(df3.head())

######################################################
# Remove null data
######################################################
display(df3.isnull().sum())
df4 = df3.dropna(subset=['乗船港コード'])
age_average = df4['年齢'].mean()
df5 = df4.fillna({'年齢': age_average})
df6 = df5.replace({'デッキ': {np.nan: 'N'}})
display(df6.isnull().sum())
display(df6.head())

######################################################
# represent the binary value data
######################################################
mf_map = {'male':1, 'female':0}
df7 = df6.copy()
df7['性別'] = df7['性別'].map(mf_map)
display(df7.head())

tf_map = {True:1, False:0}
df8 = df7.copy()
df8['成人男子'] = df8['成人男子'].map(tf_map)
display(df8.head())

df9 = df8.copy()
df9['独身'] = df9['独身'].map(tf_map)
display(df9.head())

######################################################
# represent the multi-value data
######################################################
w = pd.get_dummies(df9['男女子供'], prefix='男女子供')
display(w.head[10])

def enc(df, column):
    df_dummy = pd.get_dummies(df[column], prefix=column)
    df_drop = df.drop([column], axis=1)
    df1 = pd.concat([df_drop, df_dummy], axis=1)
    return df1

df10 = enc(df9, '男女子供')
display(df10.head())

df11 = enc(df10, '乗船港コード')
df12 = enc(df11, 'デッキ')
display(df12.head())

######################################################
# normalization/standardization
######################################################
df13 = df12.copy()
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
df13[['年齢', '料金']]=stdsc.fit_transform(df13[['年齢', '料金']])
display(df13.head())
