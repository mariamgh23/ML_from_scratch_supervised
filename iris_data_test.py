import numpy as np
import plotly.express as px

# Import your models
from Logestic_Regression import LogisticRegressionOVR
from SVM import SVM_OVR
from Simple_DecisionTree import simple_decision_tree
from Random_Forest import RandomForest_from_scratch
from XGBoost import XGBoost
from evaluation_metrics import confusion_matrix, f1_score



# 1. Load Iris Dataset

df = px.data.iris()

X = df[['sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width']].values

y = df['species'].map({
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
}).values



# 2. Train/Test Split

def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_count = int(len(X) * test_size)

    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


X_train, X_test, y_train, y_test = train_test_split(X, y)



# 3. Train Logistic OVR

log_model = LogisticRegressionOVR(lr=0.1, n_iter=2000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)



# 4. Train SVM OVR

svm_model = SVM_OVR(C=1.0, lr=0.001, n_iter=1000)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)



# 5. Train Decision Tree

tree_model = simple_decision_tree()
tree_model.fit(X_train, y_train)
y_pred_tree = np.array([tree_model.predict(x) for x in X_test])



# 6. Train Random Forest

rf_model = RandomForest_from_scratch(n_estimators=10)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)



# 7. Train XGBoost

xgb_model = XGBoost(n_rounds=10, lr=0.1)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)



# 8. Evaluation

def evaluate_model(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))


evaluate_model("Logistic Regression OVR", y_test, y_pred_log)
evaluate_model("Linear SVM OVR", y_test, y_pred_svm)
evaluate_model("Decision Tree", y_test, y_pred_tree)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb)