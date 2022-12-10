import pandas as pd
import numpy as np
import sklearn
import keras
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('diabetes.csv', header=None)
print(df.describe())

print(df.head(20))

data = df.to_numpy()
plt.subplot(3,3,1)
plt.hist(data[:,0],bins=np.arange(data[:,0].max()+1))
plt.subplot(3,3,2)
plt.hist(data[:,1],bins=np.arange(data[:,1].max()+1))
plt.subplot(3,3,3)
plt.hist(data[:,2],bins=np.arange(data[:,2].max()+1))
plt.subplot(3,3,4)
plt.hist(data[:,3],bins=np.arange(data[:,3].max()+1))
plt.subplot(3,3,5)
plt.hist(data[:,4],bins=np.arange(0,data[:,4].max()+1,10))
plt.subplot(3,3,6)
plt.hist(data[:,5],bins=np.arange(data[:,5].max()+1))
plt.subplot(3,3,7)
plt.hist(data[:,6],bins=np.arange(0,data[:,6].max()+1,0.1))
plt.subplot(3,3,8)
plt.hist(data[:,7],bins=np.arange(data[:,7].max()+1))
plt.subplot(3,3,9)
plt.hist(data[:,8],bins=np.arange(data[:,8].max()+1))
plt.tight_layout()

plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

pd.plotting.scatter_matrix(df, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

print((data==0).sum(axis=0))

df_clean = df.copy()
df_clean[[1,2,3,4,5]] = df_clean[[1,2,3,4,5]].replace(0,np.nan)
print(df_clean.isnull().sum())

df_clean = df

df_clean2 = df_clean.dropna()

df_impute = df_clean.fillna(df_clean.mean())

x_drop = df_clean2.iloc[:,0:8]
y_drop = df_clean2.iloc[:,8]

x_imp = df_impute.iloc[:,0:8]
y_imp = df_impute.iloc[:,8]

scaler = StandardScaler()
x_drop = scaler.fit_transform(x_drop)
x_imp = scaler.fit_transform(x_imp)

X_trainD, X_testD, Y_trainD, Y_testD = train_test_split(x_drop, y_drop, test_size=0.1)
X_trainI, X_testI, Y_trainI, Y_testI = train_test_split(x_imp, y_imp, test_size=0.1)

svc_drop = SVC()
svc_drop.fit(X_trainD, Y_trainD)

# Train SVC on data with imputed entries
svc_imp = SVC()
svc_imp.fit(X_trainI, Y_trainI)

accuracy_drop = svc_drop.score(X_testD, Y_testD)
print(accuracy_drop)
# Evaluate SVC on data with imputed entries
accuracy_imp = svc_imp.score(X_testI, Y_testI)
print(accuracy_imp)