import numpy as np
import plotly.express as px

# Load dataset
df = px.data.iris()

# Features
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

# Target
y = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}).values



# Binary Logistic Regression

class LogisticRegressionFromScratch:
    
    def __init__(self, lr=0.1, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        
        # Add bias
        X = np.hstack([np.ones((m, 1)), X])
        
        # Initialize weights
        self.weights = np.zeros(n + 1)

        for _ in range(self.n_iter):
            z = X @ self.weights
            h = self._sigmoid(z)
            gradient = (X.T @ (h - y)) / m
            self.weights -= self.lr * gradient

    def predict_proba(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self._sigmoid(X @ self.weights)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)



# One-vs-Rest (Multi-class)

class LogisticRegressionOVR:
    
    def __init__(self, lr=0.1, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.models = []
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models = []

        for cls in self.classes_:
            y_binary = (y == cls).astype(int)
            model = LogisticRegressionFromScratch(self.lr, self.n_iter)
            model.fit(X, y_binary)
            self.models.append(model)

    def predict(self, X):
        # Collect probabilities from each classifier
        probs = np.array(
            [model.predict_proba(X) for model in self.models]
        ).T
        
        # Get class index with highest probability
        class_indices = np.argmax(probs, axis=1)
        
        # Map index back to actual class
        return self.classes_[class_indices]