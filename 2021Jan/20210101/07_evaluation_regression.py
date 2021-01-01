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
from sklearn.datasets import load_boston
boston = load_boston()

df = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

df1 = df[['RM']]

######################################################
# Apply Regression
######################################################
from xgboost import XGBRegressor
algorithm1 = XGBRegressor(objective ='reg:squarederror',
    random_state=random_seed)
algorithm1.fit(df1, y)
y_pred1 = algorithm1.predict(df1)

from xgboost import XGBRegressor
algorithm2 = XGBRegressor(objective ='reg:squarederror',
    random_state=random_seed)
algorithm2.fit(df, y)
y_pred2 = algorithm2.predict(df)

######################################################
# Results Checking
######################################################
display(df.head())
display(df1.head())
print(y[:5])

print(f'y[:5] {y[:5]}')
print(f'y_pred1[:5] {y_pred1[:5]}')
print(f'y_pred2[:5] {y_pred2[:5]}')

######################################################
# Confirmation with Scatter Plot
######################################################
y_range = np.array([y.min(), y.max()])
print(y_range)

plt.figure(figsize=(6,6))
plt.scatter(y, y_pred1)
plt.plot(y_range, y_range, 'k--')
plt.xlabel('正解データ')
plt.ylabel('予測結果')
plt.title('正解データと予測結果の散布図表示(1入力変数)')
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y, y_pred2)
plt.plot(y_range, y_range, 'k--')
plt.xlabel('正解データ')
plt.ylabel('予測結果')
plt.title('正解データと予測結果の散布図表示(13入力変数)')
plt.show()

######################################################
# R2 Score
######################################################
from sklearn.metrics import r2_score
r2_score1 = r2_score(y, y_pred1)
print(f'R2 score(1入力変数): {r2_score1:.4f}')
r2_score2 = r2_score(y, y_pred2)
print(f'R2 score(13入力変数): {r2_score2:.4f}')
