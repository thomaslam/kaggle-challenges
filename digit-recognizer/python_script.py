import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('./train.csv')
X = df.ix[:,1:].values
Y = df['label'].values

model = LogisticRegression()
model = model.fit(X, Y)

test_df = pd.read_csv('./test.csv')
outputs = model.predict(test_df.values)
print (outputs)
