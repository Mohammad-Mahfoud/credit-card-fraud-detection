import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

credit_card_data = pd.read_csv('data/creditcard.csv')

# separate the data for analysis: 
legit = credit_card_data[credit_card_data['Class'] == 0]
fraud = credit_card_data[credit_card_data['Class'] == 1]

# under-sampling : build a sample dataset with similar distribution number of legit and fraud cases:
legit_sample = legit.sample(n=492)

# combine the legit sample with the fraud cases
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# split the data into features and target variable
X = new_dataset.drop(columns='Class', axis=1)
y = new_dataset['Class']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training:
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# training data accuracy:
X_train_prediction = model.predict(X_test)
train_data_accuracy = accuracy_score(X_train_prediction, y_test)
print(' Train Accuracy score of the logistic regression model is:', train_data_accuracy)

# testing data accuracy:
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)
print(' Test Accuracy score of the logistic regression model is:', test_data_accuracy)
