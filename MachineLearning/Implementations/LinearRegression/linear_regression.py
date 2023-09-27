import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
#print(df)

X = df[['Feature 1']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=20)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_pred = lin_model.predict(X_test)

score = lin_model.score(X_test, y_pred)
print(X_test)
print(y_pred)
print(score)

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()

# sns.regplot(x=X_test, y=y_pred, ci=None)