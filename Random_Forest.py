import numpy as np
from Simple_DecisionTree import simple_decision_tree
from collections import Counter  # <-- FIX: import Counter

# entropy and gini both calculate the impurity in classification 
# entropy E(s)=-p(+)log(P(+))-(P(-)log(p(-))) slow
#Gini =1-p(+)-p(-) Fast 
# information gain the weight of each node the heighest is the root 
# G(S,Q)=E(S)-sum(i=1,k){pi*E(S,Qi)}

class RandomForest_from_scratch:
    def __init__(self,n_estimators=5):
        self.n_estimators=n_estimators
        self.trees=[]

    def fit(self,X,y):

        for _ in range(self.n_estimators):
            idxs=np.random.choice(len(X),len(X),replace=True)
            tree=simple_decision_tree()
            tree.fit(X[idxs],y[idxs])
            self.trees.append(tree)
    

    def predict(self,X):
        predictions=np.array([[tree.predict(x) for tree in self.trees] for x in X])
        return np.array([Counter(row).most_common(1)[0][0] for row in predictions])