import pandas as pd

import pdb; pdb.set_trace()
df = pd.read_csv('5k_sample.csv')

#print(df.tail())
print(df.isnull().sum())
