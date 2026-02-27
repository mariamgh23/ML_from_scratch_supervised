import numpy as np
from Simple_DecisionTree import simple_decision_tree

class XGBoost:
    def __init__(self,n_rounds=5,lr=0.1):
        self.lr=lr
        self.n_rounds=n_rounds
        self.trees=[]
        self.classes_=[]
        self.class_to_index=None

    
    def fit(self,X,y):
        self.classes_=np.unique(y)
        self.class_to_index={c:i for i,c in enumerate(self.classes_)}
        y_indexed=np.array([self.class_to_index[c] for c in y])
        #identity matrix with the dimensions of the classes number
        y_onehot=np.eye(len(self.classes_))[y_indexed]
        # add them to the tree
        self.trees=[[] for _ in self.classes_]
        # get the dimension and fill with 1/num of classes
        preds=np.full((len(y),len(self.classes_)),fill_value=1/len(self.classes_),dtype=float)

        for _ in range(self.n_rounds):
            for k in range(len(self.classes_)):
                grad=preds[:,k]-y_onehot[:,k]
                tree=simple_decision_tree()
                # make the tree train on the mistake of prev tree
                # convert gradient to integer classes for tree compatibility
                grad_binary=(grad>0).astype(int)
                tree.fit(X,grad_binary)
                update=np.array([tree.predict(x) for x in X])
                preds[:,k]-=self.lr*update
                self.trees[k].append(tree)

    
    def predict(self,X):
        preds=np.zeros((X.shape[0],len(self.classes_)))
        for k , trees in enumerate(self.trees):
            for tree in trees:
                preds[:,k]-=self.lr*np.array([tree.predict(x)for x in X])
        return self.classes_[np.argmax(preds,axis=1)]