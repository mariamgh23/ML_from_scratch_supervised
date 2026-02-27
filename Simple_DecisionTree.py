import numpy as np

class simple_decision_tree:
    def __init__(self):
        # to know root or not 
        self.feature_index=None
        # determine left or right the branch goes
        self.threshold=None
        self.left=None
        self.right=None
        #label of class
        self.label=None

    def fit(self,X,y):
        # to stop at the leaf
        if len(set(y))==1:
            self.label=y[0]
            return
        
        best_gain=-1

        for i in range(X.shape[1]):
            thresholds=np.unique(X[:,i])
            for t in thresholds:
                left_idx=X[:,i]<=t
                right_idx=X[:,i]>t
                #skip the empty node
                if len(np.unique(y[left_idx]))<1 or len(np.unique(y[right_idx]))<1:
                    continue
                gain=self._information_gain(y,y[left_idx],y[right_idx])
                if gain>best_gain:
                    best_gain=gain
                    self.feature_index=i
                    self.threshold=t
                    self.left=simple_decision_tree()
                    self.left.fit(X[left_idx],y[left_idx])
                    self.right=simple_decision_tree()
                    self.right.fit(X[right_idx],y[right_idx])



# entropy and gini both calculate the impurity in classification 
# entropy E(s)=-p(+)log(P(+))-(P(-)log(p(-))) slow
#Gini =1-p(+)-p(-) Fast 
# information gain the weight of each node the heighest is the root 
# G(S,Q)=E(S)-sum(i=1,k){pi*E(S,Qi)}


    def gini(self,y):
        # number of frequency 
        counts=np.bincount(y)
        probs=counts/len(y)
        return 1-np.sum(probs**2) # 1-sum(p^2)
    
    def _information_gain(self,y,y_left,y_right):
        p=len(y_left)/len(y)
        return self.gini(y)-p*self.gini(y_left)-(1-p)*self.gini(y_right)
    
    def predict(self,X):
        # if X is a single sample, X must be 1D
        if self.label is not None:
            return self.label
        if X[self.feature_index]<=self.threshold:
            return self.left.predict(X)
        else:
            return self.right.predict(X)

    # optional: predict multiple samples
    def predict_batch(self, X):
        return np.array([self.predict(x) for x in X])