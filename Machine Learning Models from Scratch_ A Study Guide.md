# Machine Learning Models from Scratch: A Study Guide

## Introduction
This report summarizes the implementations of various machine learning models from scratch, based on the provided GitHub repository `ML_from_scratch_supervised`. The goal is to provide a clear and concise reference for understanding the fundamental concepts and practical implementations of these algorithms. Each model's section will cover its core principles, the implementation details found in the repository, and relevant illustrations to aid comprehension.

## Models Covered

### 1. Linear Regression

**Concept:** Linear Regression is a fundamental supervised learning algorithm used for predicting a continuous target variable based on one or more independent input features. It assumes a linear relationship between the input features and the target variable. The model aims to find the best-fitting linear equation (a line, plane, or hyperplane) that minimizes the sum of squared differences between the predicted and actual values.

**Implementation Details:** The `LinearRegressionFromScratch` class in `Linear_Regression.py` implements linear regression using gradient descent. Key aspects include:
- **Initialization:** Learning rate (`lr`) and number of iterations (`n_iter`) are set. Weights (`theta`) are initialized to zeros.
- **`fit` method:**
    - Adds a bias term to the input features `X` by concatenating a column of ones.
    - Iteratively updates the weights (`theta`) using the gradient descent optimization algorithm.
    - The gradient is calculated based on the difference between predicted and actual values (errors).
- **`predict` method:**
    - Takes new input features `X`, adds a bias term, and computes predictions using the learned weights.

**Illustration:**

graph TD
    A[Start: Initialize Weights theta=0] --> B[Add Bias Column to X]
    B --> C[Loop for n_iter]
    C --> D[Calculate Predictions: y_pred = X * theta]
    D --> E[Calculate Error: error = y_pred - y]
    E --> F[Calculate Gradient: 2/m * X.T * error]
    F --> G[Update Weights: theta = theta - lr * gradient]
    G --> C
    C --> H[End: Return Trained theta]

### 2. Logistic Regression

**Concept:** Logistic Regression is a supervised learning algorithm used for binary classification problems. Despite its name, it is a classification algorithm that models the probability of a binary outcome. It uses the sigmoid function to map the linear combination of input features to a probability between 0 and 1. A threshold (typically 0.5) is then applied to classify the outcome.

**Implementation Details:** The repository includes `LogisticRegressionFromScratch` for binary classification and `LogisticRegressionOVR` for multi-class classification using the One-vs-Rest (OvR) strategy.

**`LogisticRegressionFromScratch` (Binary):**
- **Initialization:** Learning rate (`lr`) and number of iterations (`n_iter`) are set. Weights are initialized to zeros.
- **`_sigmoid` method:** Implements the sigmoid activation function: `1 / (1 + exp(-z))`.
- **`fit` method:**
    - Adds a bias term to `X`.
    - Iteratively updates weights using gradient descent, where the gradient is derived from the difference between predicted probabilities (after sigmoid) and actual labels.
- **`predict_proba` method:** Calculates the probability of the positive class.
- **`predict` method:** Converts probabilities to binary class labels (0 or 1) based on a 0.5 threshold.

**`LogisticRegressionOVR` (Multi-class):**
- **Initialization:** Similar to binary, but also initializes a list to store individual binary logistic regression models.
- **`fit` method:**
    - Identifies unique classes in `y`.
    - For each class, it trains a `LogisticRegressionFromScratch` model to distinguish that class from all other classes (One-vs-Rest strategy).
- **`predict` method:**
    - Collects probability predictions from all individual binary models.
    - Assigns the class with the highest predicted probability as the final prediction.

**Illustration:**

graph LR
    Z[Linear Combination: z = Xw + b] --> S[Sigmoid Function: 1 / 1 + e^-z]
    S --> P[Probability: 0 to 1]
    P --> C{Threshold 0.5}
    C -->|Greater than or equal to 0.5| Class1[Class 1]
    C -->|Less than 0.5| Class0[Class 0]

### 3. Simple Decision Tree

**Concept:** A Decision Tree is a non-parametric supervised learning algorithm used for both classification and regression tasks. It works by recursively partitioning the data into subsets based on the values of input features. Each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label (for classification) or a numerical value (for regression).

**Implementation Details:** The `simple_decision_tree` class in `Simple_DecisionTree.py` implements a basic decision tree for classification.
- **Initialization:** Stores `feature_index`, `threshold`, `left` and `right` child nodes, and `label` for leaf nodes.
- **`fit` method:**
    - **Stopping Condition:** If all samples in a node belong to the same class, it becomes a leaf node with that class label.
    - **Best Split:** Iterates through all features and unique threshold values to find the split that maximizes information gain.
    - **Information Gain:** Uses the Gini impurity metric to calculate information gain. The `gini` method calculates `1 - sum(p^2)`.
    - **Recursive Fitting:** Recursively calls `fit` on the left and right child nodes.
- **`predict` method:** Traverses the tree based on feature values and thresholds until a leaf node is reached, returning its label.
- **`predict_batch` method:** Predicts for multiple samples.

**Illustration:**

graph TD
    Root[Current Node] --> Split{Find Best Split}
    Split -->|Feature i, Threshold t| Left[Left Child: X_i <= t]
    Split -->|Feature i, Threshold t| Right[Right Child: X_i > t]
    Left --> LeafL{Is Pure?}
    Right --> LeafR{Is Pure?}
    LeafL -->|Yes| LabelL[Assign Label]
    LeafL -->|No| RecL[Recursive Fit]
    LeafR -->|Yes| LabelR[Assign Label]
    LeafR -->|No| RecR[Recursive Fit]

### 4. Random Forest

**Concept:** Random Forest is an ensemble learning method for classification and regression that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees (majority vote). For regression tasks, the mean or average prediction of the individual trees is returned. It reduces overfitting and improves accuracy compared to a single decision tree.

**Implementation Details:** The `RandomForest_from_scratch` class in `Random_Forest.py` implements a random forest classifier.
- **Initialization:** Sets the number of estimators (`n_estimators`), which is the number of decision trees to build.
- **`fit` method:**
    - For each estimator, it creates a bootstrap sample (sampling with replacement) from the training data.
    - Trains a `simple_decision_tree` on each bootstrap sample.
    - Stores all trained decision trees.
- **`predict` method:**
    - For each input sample, it collects predictions from all individual decision trees.
    - Uses `collections.Counter` to perform a majority vote among the tree predictions to determine the final class label.

**Illustration:**
graph TD
    Data[Original Dataset] --> B1[Bootstrap Sample 1]
    Data --> B2[Bootstrap Sample 2]
    Data --> BN[Bootstrap Sample N]
    B1 --> T1[Decision Tree 1]
    B2 --> T2[Decision Tree 2]
    BN --> TN[Decision Tree N]
    T1 --> V[Majority Voting]
    T2 --> V
    TN --> V
    V --> Final[Final Prediction]

### 5. Support Vector Machine (SVM)

**Concept:** Support Vector Machines (SVMs) are powerful supervised learning models used for classification and regression tasks. In classification, SVMs aim to find an optimal hyperplane that best separates data points of different classes in a high-dimensional space. The optimal hyperplane is the one that maximizes the margin between the closest data points of different classes (support vectors).

**Implementation Details:** The repository provides `LinearSVM` for binary classification and `SVM_OVR` for multi-class classification.

**`LinearSVM` (Binary):**
- **Initialization:** Regularization parameter `C`, learning rate `lr`, and number of iterations `n_iter` are set. Weights `w` are initialized to zeros, and bias `b` to zero.
- **`fit` method:**
    - Iterates through the training data and updates `w` and `b` based on the hinge loss function.
    - If a data point is correctly classified and outside the margin (`y * (Xw + b) >= 1`), only regularization is applied to `w`.
    - If a data point is misclassified or inside the margin, the gradient includes the misclassification penalty.
- **`decision_function` method:** Calculates the raw score `Xw + b`.
- **`predict` method:** Returns the sign of the decision function (`-1` or `1`) for classification.

**`SVM_OVR` (Multi-class):**
- **Initialization:** Similar to `LinearSVM`, but also initializes a list to store individual binary SVM models.
- **`fit` method:**
    - Identifies unique classes.
    - For each class, it trains a `LinearSVM` model to classify that class against all others (OvR strategy), converting `y` labels to `1` or `-1`.
- **`predict` method:**
    - Collects decision scores from all individual binary SVM models.
    - The class with the highest decision score is chosen as the final prediction.

**Illustration:**

graph TD
    Start[Check Condition] --> Cond{y * Xw + b >= 1}
    Cond -->|True: Correct & Outside Margin| Update1[dw = w, db = 0]
    Cond -->|False: Inside Margin or Misclassified| Update2[dw = w - C * y * X, db = -C * y]
    Update1 --> Step[w = w - lr * dw, b = b - lr * db]
    Update2 --> Step

### 6. XGBoost

**Concept:** XGBoost (eXtreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost is known for its speed and performance, often being the algorithm of choice for structured data problems. It builds an ensemble of weak prediction models, typically decision trees, in a sequential manner, where each new tree corrects the errors of the previous ones.

**Implementation Details:** The `XGBoost` class in `XGBoost.py` implements a simplified version of the XGBoost algorithm.
- **Initialization:** Sets the number of boosting rounds (`n_rounds`) and learning rate (`lr`). Initializes lists to store trees for each class.
- **`fit` method:**
    - Converts target labels `y` to one-hot encoded format.
    - Initializes predictions with a uniform probability distribution across classes.
    - For each boosting round and for each class:
        - Calculates the gradient (residual) as the difference between current predictions and actual one-hot encoded labels.
        - Trains a `simple_decision_tree` on the input features `X` with the binarized gradient as the target (the tree learns to predict the direction of the error).
        - Updates the predictions by subtracting `lr * update` (where `update` is the tree's prediction).
        - Stores the trained tree.
- **`predict` method:**
    - Aggregates predictions from all trees for each class.
    - The class with the highest aggregated prediction score is chosen as the final prediction.



## Conclusion
This report has provided an overview of several fundamental machine learning algorithms implemented from scratch. Understanding these implementations offers valuable insights into the inner workings of these powerful models, from the iterative optimization of Linear and Logistic Regression to the ensemble power of Random Forests and the error-correcting nature of XGBoost. The provided illustrations aim to further clarify the core mechanisms of each algorithm, serving as a useful study reference.
