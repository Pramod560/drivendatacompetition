# -*- coding: utf-8 -*-
"""dengAI.py

dengue prediction using past data
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
# %matplotlib inline

full_train_data = pd.read_csv("dengue_features_train.csv")
full_train_target= pd.read_csv("dengue_labels_train.csv")
full_test_data = pd.read_csv("dengue_features_test.csv")
full_test_target = pd.read_csv("submission_format.csv")
train_data = full_train_data.drop(["year","weekofyear","week_start_date"],axis = 1)
train_target = full_train_target["total_cases"]
test_data = full_test_data.drop(["year","weekofyear","week_start_date"],axis = 1)

plt.style.use("fivethirtyeight")
plt.bar(full_train_data["weekofyear"],train_target)
plt.xlabel("weeks")
plt.ylabel("total cases")
plt.title("dengue cases prediction")
plt.show()

def clean_data(data):
    data.loc[data["city"] == 'sj', "city"] = 0
    data.loc[data["city"] == "iq", "city"] = 1
def handle_NaN(data):
    data.fillna(data.mean(), inplace = True)

clean_data(train_data)
handle_NaN(train_data)
clean_data(test_data)
handle_NaN(test_data)
display(train_data.head())

missing_val_count_by_column = (train_data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

"""min_max_scaler = MinMaxScaler()
minmax_train_data = min_max_scaler.fit_transform(train_data)
minmax_test_data = min_max_scaler.fit_transform(test_data)"""

x_train, x_test, y_train, y_test = train_test_split(train_data, train_target, test_size = 0.25)

clf = LinearRegression()
clf.fit(x_train, y_train)
y = clf.predict(x_test)

print(mean_absolute_error(y_test, y))
print(mean_squared_error(y_test, y))
print(r2_score(y_test, y))

clf1 = RandomForestRegressor()
clf1.fit(x_train, y_train)
y1 = clf1.predict(x_test)

print(mean_absolute_error(y_test, y1))
print(mean_squared_error(y_test, y1))
print(r2_score(y_test, y1))

clf1 = SGDRegressor(max_iter=1000, tol = 1e-3)
clf1.fit(x_train, y_train)
y1 = clf1.predict(x_test)

print(mean_absolute_error(y_test, y1))
print(mean_squared_error(y_test, y1))
print(r2_score(y_test, y1))

clf1 = SVR(kernel = "rbf",degree=3, gamma="scale")
clf1.fit(x_train, y_train)
y1 = clf1.predict(x_test)

print(mean_absolute_error(y_test, y1))
print(mean_squared_error(y_test, y1))
print(r2_score(y_test, y1))

clf1 = DecisionTreeRegressor()
cross_val_score(clf1, train_data, train_target, cv = 10)

display(pd.DataFrame([y1, y_test]))

y2 = clf1.predict(minmax_test_data)

