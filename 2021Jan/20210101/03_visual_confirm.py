#####################################################
# Configuration
#####################################################
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

#####################################################
# Read Data
#####################################################
import seaborn as sns

df_iris = sns.load_dataset("iris")

column_i = ['萼片長', '萼片幅', '花弁長', '花弁幅', '種別']
df_iris.columns= columns_i

display(df_iris.head())

#####################################################
# Display data with scatter plot (matplotlib)
#####################################################
plt.figure(figsize=(6, 6))

plt.scatter(df_iris['萼片幅'], df_iris['花弁長'])
plt.xlabel('萼片幅')
plt.ylabel('花弁長')
plt.show()

#####################################################
# Display data with scatter plot (seaborn)
#####################################################
plt.figure(figsize=(6, 6))

sns.scatterplot(x='萼片幅', y='花弁長', hue='種別', s=70, data=df_iris)
plt.show()

#####################################################
# Display data with scatter plot (pairplot)
#####################################################
sns.pairplot(df_iris, hue="種別")
plt.show()

#####################################################
# Display data with scatter plot (jointplot)
#####################################################
sns.jointplot('萼片幅', '花弁長', data=df_iris)
plt.show()

#####################################################
# Display data with box plot (data frame)
#####################################################
plt.figure(figsize=(6, 6))

df_iris.boxplot(patch_artist=True)
plt.show()

#####################################################
# Display data with box plot (seaborn)
#####################################################
w = pd.melt(df_iris, id_vars=['種別'])

display(w.head())
plt.figure(figsize=(8, 8))
sns.boxplot(x="variable", y="value", data=w, hue="種別")
plt.show()
