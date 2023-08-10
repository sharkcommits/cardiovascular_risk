#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

data = pd.read_csv('CVD_cleaned.csv')

#Get the data summary.
summary = data.describe()
print(summary)

#Find whether there are missing values or not.
print(data.isnull().describe())

#Data Cleaning Part.
"""
Since BMI tells us the ratio between height and weight, we can safely remove 
height and weight columns and use the BMI values in advance.
"""
data = data.drop(['Height_(cm)', 'Weight_(kg)'], axis=1)

#List of the Categorical Data:
list_of_categories = ["General_Health", "Checkup", "Exercise", "Heart_Disease", "Skin_Cancer",
"Other_Cancer", "Depression", "Diabetes", "Arthritis", "Sex", "Age_Category",
"Smoking_History"]

#Then we transform our categorical data into new encoded data.
def transform_categorical_columns(df, categorical):
    ohe = OneHotEncoder(sparse_output=False)
    le = LabelEncoder()
    empty_dataframe = pd.DataFrame()
    for i in categorical:
        #If it just contains two variables, then it is best to use LE.
        if df[i].nunique() == 2:
            temporary = pd.DataFrame(le.fit_transform(np.array(df[i])), columns=[i])
            empty_dataframe = pd.concat([empty_dataframe, temporary], axis=1)
            df.drop([i], axis=1, inplace=True)
        #Else, it's OHE.    
        else:
            temporary = pd.DataFrame(ohe.fit_transform(np.array(df[i]).reshape(-1, 1)))
            temporary.columns = ohe.get_feature_names_out([i])
            empty_dataframe = pd.concat([empty_dataframe, temporary], axis=1)
            df.drop([i], axis=1, inplace=True)
    return pd.concat([empty_dataframe, df], axis=1)

#Our encoded data:
final_dataframe = transform_categorical_columns(data, list_of_categories)

#We seperate our x and y.
y = final_dataframe['Heart_Disease']
x = final_dataframe.drop(['Heart_Disease'], axis=1)

#This part is to see whether the classification data balanced or imbalanced...
value_zero = final_dataframe[final_dataframe['Heart_Disease'] == 0]
value_one = final_dataframe[final_dataframe['Heart_Disease'] == 1]

#...which is imbalanced as we can see. Majority class is much less 10 times higher.
print(value_zero.shape, value_one.shape)

#Then we use SMOTE technique to populate our minority class by KNN, not just duplicate them.
smote = SMOTE(sampling_strategy='minority')

#X_sy and y_sy are our new final form of data and THEY'RE BALANCED.
X_sy, y_sy = smote.fit_resample(x, y)
"""
This part is just for GridSearchCV usage.

params = [{"loss": ['log_loss', 'hinge']},
          {"penalty": ['elasticnet']},
          {"max_iter": [1000, 2000]}]
"""

#Our data is just getting prepared and ready to be trained.
X_train, X_test, y_train, y_test = train_test_split(X_sy, y_sy, test_size=0.2, random_state=31)

xgb = XGBClassifier()

"Using KFold with our populated data."
kfold = KFold(n_splits=10)
results = cross_val_score(xgb, X_sy, y_sy, cv=kfold)
print("Accuracy(with SMOTE and 10K-Fold): ", results.mean())

"Using Stratified K-Fold with our imbalanced data, no SMOTE."

skfold = StratifiedKFold(n_splits=10)
res = cross_val_score(xgb, x, y, cv=skfold)
print("Accuracy(with Stratified 10K-Fold): ", res.mean())













