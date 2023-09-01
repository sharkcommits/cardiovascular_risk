# Cardiovascular Risk Prediction Model

### This model aims to predict a person's cardiovascular risk based on given features such as BMI, Smoking History, Age Category, Fruit Consumption etc.

## Table of Contents

- Installation
- Motivation behind the Project
- File Description
- Results
- Licensing, Authors, Acknowledgements

## Installation

The code requires Python 3 and general libraries available through the Anaconda package.

## Motivation behind the Project

It is vital to have some knowledge about how much risk does a person have when it comes to health.
We aim to have better knowledge about the risk behind the everyday life, habits and general body condition and 
help people by providing general understanding which the model has.

To find the results we use:

- XGBClassifier with SMOTE
- Stratified K-Fold (for imbalanced data)

We're gonna compare the results and decide which one to go.

## File Description

This project includes one Jupyter Notebook with all code required for analyzing the data and creating a supervised 
machine learning algorith. The csv file containes 308854 people along with the following features:

- General_Health (Condition)
- Checkup (within years)
- Exercise (y/n)
- Skin_Cancer (y/n)
- Other_Cancer (y/n)
- Depression (y/n)
- Diabetes (y/n)
- Arthritis (y/n)
- Sex (f/m)
- Age_Category
- Height
- Weight
- BMI
- Smoking_History
- Alcohol_Consumption
- Fruit_Consumption
- Green_Vegetables_Consumption
- FriedPotato_Consumption

## Results

We get **94.57%** accuracy by using SMOTE and 10K-Fold and **91.91%** by using Stratified 10K-Fold.

## Licensing, Authors and Acknowledgements

This dataset comes from [Kaggle Datasets](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset) The License
is [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).