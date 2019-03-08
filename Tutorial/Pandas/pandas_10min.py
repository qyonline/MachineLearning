"""

http://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html

"""
import pandas as pd
import numpy as np
#Creating a DataFrame by passing a dict of objects that can be converted to series-like.
df = pd.DataFrame({'A': 1.,
                   'B': pd.Timestamp('20130102'),
                   'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                   'D': np.array([1,2,3,4], dtype='int32'),
                   'E': pd.Categorical(["test", "train", "test", "train"]),
                   'F': 'foo'})

print(df)

print(df.loc[[1,3], ['A', 'D']])
print(df.loc[1:3, ['A', 'D']])
print(df.iloc[3]) #type of iloc is <class 'pandas.core.series.Series'>
