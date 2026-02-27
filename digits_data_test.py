from sklearn.datasets import load_digits
import pandas as pd
import numpy as np 
from Logestic_Regression import LogisticRegressionOVR
from SVM import SVM_OVR
from evaluation_metrics import confusion_matrix,f1_score

digits=load_digits()
X=digits.data
y=digits.target

df=pd.DataFrame(X)
df["target"]=y

# train test split 
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    test_count = int(len(X) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y)

#Logestic OVR
print("Training Logistic Regression OVR...")
log_model = LogisticRegressionOVR(lr=0.1, n_iter=3000)  # increase iterations for digits
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

#linear SVM OVR
print("Training Linear SVM OVR...")
svm_model = SVM_OVR(C=1.0, lr=0.0005, n_iter=2000)  # smaller lr due to higher features
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# evaluation
print("\n===== Logistic Regression OVR =====")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))
print("F1 Score:", f1_score(y_test, y_pred_log))

print("\n===== Linear SVM OVR =====")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print("F1 Score:", f1_score(y_test, y_pred_svm))
