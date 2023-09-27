import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('simplelinearregression.csv')
print(data)

X_train, X_test, y_train, y_test = train_test_split(data[['Age']],data['Premium'],test_size=0.25, random_state=25)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
Lin_Regression = LinearRegression()
Lin_Regression.fit(X_train, y_train)

y_pred = Lin_Regression.predict(X_test)
print(y_pred)
print(y_test)
plt.scatter(X_test, y_test, color = 'black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()
