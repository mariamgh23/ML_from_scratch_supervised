import numpy as np 
def gradient_descent(X,y,lr=0.1,n_iter=1000):
    #MSE rule and gradient rule in mind , linear equation
    # rows and columns m=> rows , n=> columns
    m,n=X.shape
    # horizontal stack => to make matrix with adding bias by adding ones by the number of sample
    X=np.hstack([np.ones((m,1)),X])
    # the weight= columns let's say 5 [0,0,0,0,0] +1 the intersect
    theta=np.zeros(n+1) 
    # fixed iteration 
    for _ in range(n_iter):
        y_pred=X @ theta  # ypred=W.X
        # not MSE because we want to get the sum in the loop
        error=y_pred-y
        # grad_loss=1/m((x^T).(ypred-y))
        gradient=X.T @ error/m
        #w new=w old-lr * gradient
        theta-=lr*gradient
    return theta 

