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
df2 = df[['Month of the year' , 'Day of the month','Hour']]
#transactions = df.Transactions
#print(transactions.describe())

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
