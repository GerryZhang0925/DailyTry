###################################################
# Configuration
###################################################
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

###################################################
# Read data
###################################################
import seaborn as sns

df_titanic = sns.load_dataset("titanic")

columns_t = ['生存', '等室', '性別', '年齢', '兄弟配偶者数',
             '両親子供数', '料金', '乗船港コード', '等室名',
             '男女子供', '成人男子', 'デッキ', '乗船港', '生存可否', '独身']
df_titanic.columns = columns_t
display(df_titanic.head())

###################################################
# Find the data which is invalid
###################################################
print(df_titanic.isnull().sum())

print(df_titanic['乗船港'].value_counts())
print(df_titanic['生存可否'].value_counts())

###################################################
# Investigate statistically
###################################################
display(df_titanic.describe())
display(df_titanic.groupby('性別').mean())

###################################################
# Display the numeric data
###################################################
columns_n = ['生存', '等室', '年齢', '兄弟配偶者数', '両親子供数', '料金']
plt.rcParams['figure.figsize'] = (10, 10)
df_titanic[columns_n].hist()
plt.show()

###################################################
# Display the non-numeric data
###################################################
columns_c = ['性別', '乗船港', '等室名', '成人男子']
plt.rcParams['figure.figsize'] = (8, 8)
for i, name in enumerate(columns_c):
    ax = plt.subplot(2, 2, i+1)
    df_titanic[name].value_counts().plot(kind='bar', title=name, ax=ax)

plt.tight_layout()
plt.show()

    
