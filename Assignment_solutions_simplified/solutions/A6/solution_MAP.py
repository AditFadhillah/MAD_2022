import sys
sys.path.append("../code/")

import numpy as np
import matplotlib.pyplot as plt
import A6_Bayes as aux

# Load data set from CSV file
t, X = aux.loaddata('../data/binary_classes.csv')
#t, X = aux.loaddata('binary_classes.csv')


# Visualize
row0 = (t == 0)
row1 = (t == 1)

plt.figure()
plt.plot(X[row0,0], X[row0,1], 'b.')
plt.plot(X[row1,0], X[row1,1], 'r*')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')




# Ex3 a)
def gradient(t, X, w, sigma2, P):
    """Assumes sigma2 = sigma**2"""
    return np.sum(X*(t.reshape((t.shape[0],1))-P), axis=0).reshape((2,1)) - w / sigma2
    
def Hessian(t, X, w, sigma2, P):
    """Assumes sigma2 = sigma**2"""
    XX= np.tensordot(X,X.T, axes=0) * (P*(1-P)).reshape((X.shape[0], 1, 1, 1))
    return -np.eye(2) / sigma2 - np.einsum('ijki->jk', XX)
    

def MAP(t, X, w0, sigma2):
    """Implements the Newton-Raphson optimization method. 
    Returns the MAP point estimate for w."""
    
    w = w0
    stopping = False
    max_iter = 10
    iter = 0
    # Iterate
    while not stopping:
        wold = w
        iter += 1
        P = aux.sigmoid(X,w)
        gradw = gradient(t, X, w, sigma2, P)
        H = Hessian(t, X, w, sigma2, P)
        invH = np.linalg.inv(H)
        w = w - np.dot(invH, gradw) 
        
        
        if np.linalg.norm(w-wold) < np.finfo(float).eps or iter > max_iter:
            print("stopping at iter = " + str(iter) + ", weight differences = " + str(np.linalg.norm(w-wold)))
            stopping = True
        else:
            print("iter = " + str(iter))
    
    return w

    
# Ex3 b)
w0=np.zeros((2,1))
#w0=np.ones((2,1))
sigma2 = 10

wmap = MAP(t, X, w0, sigma2)


print(wmap)

# Visualize
row0 = (t == 0)
row1 = (t == 1)

plt.figure()
plt.plot(X[row0,0], X[row0,1], 'b.')
plt.plot(X[row1,0], X[row1,1], 'r*')
plt.plot([-4, 4], [wmap[1] / wmap[0] * 4, -wmap[1] / wmap[0] * 4],'g')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Decision boundary')


# Ex3 c)

aux.contourplot(t, X, wmap, aux.sigmoid)



plt.show()

