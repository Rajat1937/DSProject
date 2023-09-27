import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics

cancer = datasets.load_breast_cancer()

print('Features :',cancer.feature_names)
print('Labels :', cancer.target_names)

print('Shape :', cancer.data.shape)

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.33, random_state=29)

svc_model = SVC(kernel='linear')
svc_model.fit(X_train, y_train)

y_pred = svc_model.predict(X_test)

print('Accuracy :', metrics.accuracy_score(y_test, y_pred))
print('Report :', metrics.classification_report(y_test, y_pred))

y_train_pred = svc_model.decision_function(X_train)    
y_test_pred = svc_model.decision_function(X_test) 

train_fpr, train_tpr, tr_thresholds = metrics.roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = metrics.roc_curve(y_test, y_test_pred)

plt.grid()

plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(metrics.auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(metrics.auc(test_fpr, test_tpr)))
plt.plot([0,1],[0,1],'g--')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC(ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()
# fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
# plt.legend(loc=4)
# plt.show()