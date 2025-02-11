import numpy as np
import matplotlib.pyplot as plt


def loaddata(filename):
    """Load data from filename and return t, X
        t - N-dim. vector of target values
        X - (Nx2) matrix containing the input vectors X for each data point
    """
    # Load data set from CSV file
    Xt = np.loadtxt(filename, delimiter=',')

    # Split into data matrix and target vector
    X = Xt[:,0:2]
    t = Xt[:,2]
    
    return t, X



def sigmoid(X, w):
    """Returns the probability of the class t=1, P(T_{new} = 1 | \vec{x}_{new}, \hat{\vec{w}}). 
    Assumes 
    - X is a (NxD) matrix , 
    - w is a (Dx1) vector of model parameters.
    returns a (Nx1) vector of the probabilities"""
    return 1/(1+np.exp(-np.dot(X,w)))



def contourplot(t, X, w, prob):
    """Makes a contour plot of the probability function together with the data points (blue for t=0, and red for t=1).
        t - N-dim. vector of target values
        X - (Nx2) matrix containing the input vectors X for each data point
        w - parameters, either a (2x1) vector or a matrix of parameter vector samples
        prob - the function to plot contours for given by a function name to be called with prob(X, w) and must return a M-vector of probabilities
               (sigmoid as an example).
    """
    delta = 0.025
    x = np.arange(-4.0, 4.0, delta)
    y = np.arange(-4.0, 4.0, delta)
    XX, YY = np.meshgrid(x, y)
    
    P = prob(np.hstack([XX.reshape((np.prod(XX.shape),1)), YY.reshape((np.prod(YY.shape),1))]) , w).reshape(XX.shape)
    
    fig, ax = plt.subplots()
    CS = ax.contour(XX, YY, P)
    ax.clabel(CS, fontsize=9, inline=1)
    
    # Plot the data on top
    row0 = (t == 0)
    row1 = (t == 1)
    ax.plot(X[row0,0], X[row0,1], 'b.')
    ax.plot(X[row1,0], X[row1,1], 'r*')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')



# Test code
if (__name__=='__main__'):
    # Load data
    t, X = loaddata('../data/binary_classes.csv')
    
    # Pick some parameter values
    w=np.ones((2,1))
    
    # Make plot using sigmoid function
    contourplot(t, X, w, sigmoid)

    plt.show()
    