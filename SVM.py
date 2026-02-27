# Linear SVM binary classification
import numpy as np
import plotly.express as px
from gradient_descent import gradient_descent
# load data
df = px.data.iris()
# split data features and target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}).values

# lagrange(w,b,c)=1/2(|W|^2 + sum(i=1,n){Ci(1-yi(W.Xi+b))})


class LinearSVM:
    
    def __init__(self, C=1.0, lr=0.001, n_iter=1000):
        self.C = C
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = 0

    def fit(self, X, y):
        m, n = X.shape
        
        # Expect y already in {-1,1}
        self.w = np.zeros(n)
        self.b = 0

        for _ in range(self.n_iter):
            for i in range(m):
                condition = y[i] * (np.dot(X[i], self.w) + self.b) >= 1

                if condition:
                    dw = self.w
                    db = 0
                else:
                    dw = self.w - self.C * y[i] * X[i]
                    db = -self.C * y[i]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))
    
# more that two classes
class SVM_OVR:
    
    def __init__(self, C=1.0, lr=0.001, n_iter=1000):
        self.models = []
        self.classes_ = []
        self.C = C
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models = []

        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, -1)
            model = LinearSVM(C=self.C, lr=self.lr, n_iter=self.n_iter)
            model.fit(X, y_binary)
            self.models.append(model)

    def predict(self, X):
        decision_scores = np.array(
            [model.decision_function(X) for model in self.models]
        ).T
        
        class_indices = np.argmax(decision_scores, axis=1)
        return self.classes_[class_indices]

        





       
