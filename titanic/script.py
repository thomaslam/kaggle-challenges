import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('./data/train.csv')
df = df.drop(['Cabin', 'Name', 'Ticket'], axis = 1)

# fill na
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# make gender and port columns of data type int
df["Gender"] = df["Sex"].map({'female': 0, 'male': 1}).astype(int)
df["Port"] = df["Embarked"].map({'C': 1, 'S': 2, 'Q': 3}).astype(int)

df = df.drop(['Sex', 'Embarked'], axis=1)

# switch survived column to leftmost
cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]
# print(df.head(10))

# sciki-learn: train the model using randomforest
train_data = df.values

model = RandomForestClassifier(n_estimators = 100)
model = model.fit(train_data[:, 2:], train_data[:, 0])

# similar data processing as training data
df_test = pd.read_csv('./data/test.csv')
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis = 1)

df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())
df_test['Embarked'] = df_test['Embarked'].fillna(df_test['Embarked'].mode()[0])

df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male':1})
df_test['Port'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3})

df_test = df_test.drop(['Sex', 'Embarked'], axis=1)

# Fill NaN values in Fare column with mean of Fare based on column Pclass
fare_means = df_test.pivot_table('Fare', index='Pclass', aggfunc='mean')
# print(fare_means) should return this
# Pclass
# 1    94.280297
# 2    22.202104
# 3    12.459678
# Name: Fare, dtype: float64
df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:
                            fare_means[x['Pclass']] if pd.isnull(x['Fare'])
                            else x['Fare'], axis=1)

test_data = df_test.values
output = model.predict(test_data[:, 1:])
result = np.c_[test_data[:, 0].astype(int), output.astype(int)]
df_result = pd.DataFrame(result[:, 0:2], columns = ['PassengerId', 'Survived'])
df_result.to_csv('./titanic_result.csv', index=False)
