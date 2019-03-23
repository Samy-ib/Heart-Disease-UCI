#%%
import pandas as pd
import numpy as np

'''
    Randomize the dataset rows.
'''


df = pd.read_csv('./data/heart.csv')

#%%
df.head()

#%%
df = df.sample(frac=1)
df
df.to_csv("./data/heart_processed.csv", index=False)
#%%
df['age'].value_counts().sort_index()

#%%
df.loc[(df.age<=40) & (df.age>=29), 'age'] = 1
df.loc[(df.age<=44) & (df.age>=41), 'age'] = 2
df.loc[(df.age<=50) & (df.age>=45), 'age'] = 3
df.loc[(df.age<=55) & (df.age>=51), 'age'] = 4
df.loc[(df.age<=60) & (df.age>=56), 'age'] = 5
df.loc[(df.age<=65) & (df.age>=61), 'age'] = 6
df.loc[(df.age<=77) & (df.age>=66), 'age'] = 7

df