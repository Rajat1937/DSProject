import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('data.csv')

X = data[['Feature 1', 'Feature 2', 'Feature 3']]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

y_pred = multi_model.predict(X_test)

print('Mean Square Error :', mean_squared_error(y_test, y_pred))
print('Mean Absolute Error :', mean_absolute_error(y_test, y_pred))