#!/usr/bin/env python
# coding: utf-8

# ### Exercise 2


# Loading packages
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import matplotlib
matplotlib.rcParams['figure.figsize'] = (15,7)
matplotlib.rc('font', size=15)
matplotlib.rc('axes', titlesize=15)



def visualize_model(mu, Sigma, xmin, xmax, ymin, ymax, x, t, Nsample):
    """This function visualize a normal distribution of linear model parameter (w_0, w_1) by a contour plot
    and by plotting the data points and random samples of parameters (w_0, w_1) as lines.
    
    As input it takes the following parameters:
    mu - the mean (2x1) of the normal distribution in parameter space (w_0, w_1)
    Sigma - the covariance matrix (2x2) of the normal distribution in parameter space (w_0, w_1)
    xmin, xmax, ymin, ymax - the ranges of (w_0, w_1) to use when plotting the contour plot
    x - the input vector of years
    t - the target vector of running times
    Nsample - number of random samples to illustrate (each is plotted as a line)
    """
    # First, we visualize the model by visualizing the 
    # prior/posterior normal distribution over the model parameters $[w_0, w_1]^T$.
    
    # Define a grid for visualizing normal distribution and define the normal distribution on the grid
    xx, yy = np.mgrid[xmin:xmax:.01, ymin:ymax:.01]
    pos = np.dstack((xx, yy))
    rv = multivariate_normal(mu.flatten(), Sigma)    
    
    # Plot the normal distribution
    fig, ax = plt.subplots(1,2)
    ax[0].contourf(xx, yy, rv.pdf(pos))
    ax[0].set_xlabel('$w_0$')
    ax[0].set_ylabel('$w_1$')
    ax[0].set_title('Distribution over model parameters $[w_0, w_1]^T$')

    # Second, we visualize the model by drawing samples from the prior/posterior and 
    # visualizing the corresponding regression lines
    
    # First, we scatter plot the observed data
    ax[1].scatter(x, t)
    
    # draw sample model parameters from the model    
    w_0, w_1 = np.random.multivariate_normal(mu.flatten(), Sigma, Nsample).T
    
    # Plot the corresponding sample regression lines
    for i in range(Nsample):
        ax[1].plot([0, 30], [w_0[i] + 0*w_1[i], w_0[i] + 30*w_1[i]])    
    
    ax[1].set_xlabel('x, Year')
    ax[1].set_ylabel('t, Time')    
    ax[1].set_title('Sample regression lines from the model')
    ax[1].set_xlim(0,30)
    ax[1].set_ylim(8,13)


# Loading data
data = np.loadtxt('men-olympics-100.txt')
N, d = data.shape
print('N = ', N)
print('d = ', d)

x = (data[:,0]-data[0,0]).reshape(N,1)  / 4 # Shift and rescale the input for visualization purposes
t = data[:,1].reshape(N,1)
one = np.ones((N,1))
X = np.concatenate((one, x), axis = 1)

plt.scatter(x,t)
plt.xlabel('x')
plt.ylabel('t')
plt.title('The olympic 100 dataset')


# Set parameters
sigma = np.sqrt(10)
mu0 = np.array([0, 0]).reshape(2,1)
Sigma0 = np.array([[100, 0],[0, 5]])
print("sigma = " + str(sigma))
print("mu0 = " + str(mu0))
print("Sigma0 = " + str(Sigma0))


# **Exercise 2c)**

# Implement posterior
def posterior(X, y, mu0, Sigma0):
    # X = data matrix
    # mu0, Sigma0 = mean and covariance of prior normal distribution
    invSigma0 = np.linalg.inv(Sigma0)
    Sigmaw = np.linalg.inv(np.matmul(X.T, X)/(sigma**2) + invSigma0)
    #test = np.matmul(invSigma0, mu0)
    muw = np.matmul(Sigmaw, (np.matmul(X.T,y)/(sigma**2)+np.matmul(invSigma0, mu0)))
    
    return muw, Sigmaw


# **Exercise 2d)**


# Printing mean and covariance of posterior density
muw, Sigmaw = posterior(X,t,mu0, Sigma0)
print("muw = " + str(muw))
print("Sigmaw = " + str(Sigmaw))


        
    
# Call the function with your predictions of muw and Sigmaw
visualize_model(muw, Sigmaw, 7,15,-0.3,0.3,x,t, 10)


# Plot the line corresponding to the posterior mean
plt.figure()
plt.scatter(x,t)
plt.plot([0,30], [muw[0], muw[0] + 30*muw[1]],'r')


plt.show()



