import numpy as np
import plotly.express as px
from gradient_descent import gradient_descent
# load data
df = px.data.iris()
# split data features and target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}).values


class LinearRegressionFromScratch:
    
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.theta = None

    def fit(self, X, y):
        m, n = X.shape
        
        # Add bias column
        X_bias = np.hstack([np.ones((m, 1)), X])
        
        # Initialize weights
        self.theta = np.zeros(n + 1)

        for _ in range(self.n_iter):
            predictions = X_bias @ self.theta
            errors = predictions - y
            gradients = (2/m) * (X_bias.T @ errors)
            self.theta -= self.lr * gradients

    def predict(self, X):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_bias @ self.theta

    


        

