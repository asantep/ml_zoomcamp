#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# Parameters
C = 1.0
n_splits = 5

output_file = f'model_C={C}.bin'

df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/datasets/master/AER_credit_card_data.csv')
df.columns = df.columns.str.lower()
df.head()


# CREATING THE TARGET VARIABLE:
card_values = {
    "yes": 1,
    "no": 0
}
df["card"] = df.card.map(card_values)
df.head(10)


# INITIALIZING NUMERICAL AND CATEGORICAL VARIABLES:
numerical = ["reports", "age", "income", "share", "expenditure", "dependents", "months", "majorcards", "active"]
categorical = ["owner", "selfemp"]

# SPLITTING THE DATASET

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# # TRAINING LOGISTIC REGRESSION MODEL

columns = categorical + numerical


def train(df_train, y_train, C=1.0):
    dicts = df_train[columns].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model


def predict(df, dv, model):
    dicts = df[columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


print(f'doing validation with C={C}')
# INITIALIZING KFOLD CROSS VALIDATION:
scores = []

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.card
    y_val = df_val.card

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)


print('Results:')
print(f'C={C}, mean={np.mean(scores)} , std={np.std(scores)}')


print('training the final model')
dv, model = train(df_full_train, df_full_train.card.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.card.values
auc = roc_auc_score(y_test, y_pred)
print(f' auc is {auc}')

# Save Model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')