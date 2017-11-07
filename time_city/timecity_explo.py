import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

df = pd.read_csv('1.csv')
df2 = df[['Device Category','Month of the year', 'Day of the month','Hour']]
#transactions = df.Transactions
#print(transactions.describe())
print(df2.head(40))

#import pdb;pdb.set_trace()
platform_mapping = {"mobile":0,"tablet":1,"desktop":1}

df2['Device Category'] = df2['Device Category'].map(platform_mapping)

print(df2.head(40))

target = df.Transactions
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, df2, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

#print(df.head())
#print(df.shape)


clf = KNeighborsClassifier(n_neighbors=45)
scoring = 'accuracy'
score = cross_val_score(clf, df2, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
