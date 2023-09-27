import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

#col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
data = pd.read_csv('diabetes.csv')
data = data.rename(columns={'Pregnancies':'pregnant', 'Glucose':'glucose','BloodPressure':'bp', 'SkinThickness':'skin',
                            'Insulin':'insulin', 'BMI':'bmi','DiabetesPedigreeFunction':'pedigree','Age':'age', 'Outcome':'label'})
feature_cols = ['glucose', 'bp', 'insulin', 'bmi', 'pedigree', 'age']

X = data[feature_cols]
y = data.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

# print(data)
Log_model = LogisticRegression(random_state=16, max_iter=1000)
Log_model.fit(X_train, y_train)

y_pred = Log_model.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix :', cnf_matrix)
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
target_names = ['without diabetes', 'with diabetes']
print(classification_report(y_test, y_pred, target_names=target_names))

import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

y_pred_proba = Log_model.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
