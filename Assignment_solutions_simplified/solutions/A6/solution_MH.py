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


# Ex4, a)
def sample_prior(mu_prior, sigma_square, rng):
    """Returns a sample from the prior"""
    cov = sigma_square * np.eye(mu_prior.shape[0])
    return rng.multivariate_normal(mu_prior, cov, 1).reshape((mu_prior.shape[0],))


def generate_proposal(w_old, rng):
    """Samples a proposal w from the proposal distribution based on old value w_old.
       The proposal distribution is a Gaussian distribution centered in w_old."""
    sigma_square = 0.5 # 0.01, 0.05, 0.1, 0.5, 1.0
    cov = sigma_square * np.eye(w_old.shape[0])
    return rng.multivariate_normal(w_old, cov, 1).reshape((w_old.shape[0],))
    


def prior(w, sigma2):
    """Function that computes the prior value for
    w - a (Dx1) vector
    sigma2 - sigma^2 parameter
    returns the prior density value
    """
    D = w.shape[0]
    return np.exp(-np.dot(w.T,w) / (2.0*sigma2)) / (np.sqrt(2 * np.pi * sigma2))**D


def log_prior(w, sigma2):
    """Function that computes the prior value for
    w - a (Dx1) vector
    sigma2 - sigma^2 parameter
    returns the prior density value
    """
    D = w.shape[0]
    return -np.dot(w.T,w) / (2.0*sigma2) - D / 2.0 * np.log(2 * np.pi) - D / 2.0 * np.log(sigma2)

def class0(X, w):
    return np.exp(-np.dot(X,w)) / (1+np.exp(-np.dot(X,w)))

def likelihood(t, X, w):
    P1 = aux.sigmoid(X, w)
    P0 = class0(X, w)
    return np.prod(P1**(t.reshape((t.shape[0],))) * P0**(1-t.reshape((t.shape[0],))))


def log_class0(X, w):
    return -np.dot(X, w) - np.log(1 + np.exp(-np.dot(X, w)))

def log_class1(X, w):
    return - np.log(1 + np.exp(-np.dot(X, w)))

def log_likelihood(t, X, w):
    P1 = log_class1(X, w) # Not really necessary, could have used log(sigmoid)
    P0 = log_class0(X, w)
    return np.sum( P1*(t.reshape((t.shape[0],))) + P0*(1-t.reshape((t.shape[0],))) )

    
def acceptance_ratio(wnew, wold, t, X, sigma2):
    return prior(wnew, sigma2) * likelihood(t, X, wnew) / (prior(wold, sigma2) * likelihood(t, X, wold))
    

def log_acceptance_ratio(wnew, wold, t, X, sigma2):
    return log_prior(wnew, sigma2) + log_likelihood(t, X, wnew) - (log_prior(wold, sigma2) + log_likelihood(t, X, wold))


    
def metropolis_hastings(t, X, sigma2, N_samples):
    """Metropolis-Hastings algorithm applied to the binary response model.
       Returns an array of D x N_samples samples."""

    # Allocate memory for the output array
    samples = np.zeros((X.shape[1], N_samples))
    copies = np.zeros((N_samples,))

    # Initialize random number generator
    rng = np.random.default_rng()

    # Initial sample drawn from prior
    #w_current = sample_prior(np.zeros((2,)), sigma2, rng)
    #w_current = np.ones((2,))
    #w_current = np.zeros((2,))
    #w_current =np.array([1.11604438, 2.58042458]).reshape((2,))
    #w_current = np.array([0.5, 0.5]).reshape((2,))
    #w_current = np.array([1.67375445, 1.98760995]).reshape((2,))
    w_current = np.array([2.2, 2.5]).reshape((2,))
    

    for s in range(0, N_samples):

        # Generate a valid sample from the proposal distribution
        w_proposal = generate_proposal(w_current, rng)

        r = acceptance_ratio(w_proposal, w_current, t, X, sigma2)
        #r = log_acceptance_ratio(w_proposal, w_current, t, X, sigma2)
        if r >= 1.0: # acceptance_ratio
        #if r >= 0.0: # log_acceptance_ratio
            samples[:, s] = w_proposal
            w_current = w_proposal
        else:
            u = rng.uniform(0.0, 1.0, 1)
            if (u <= r): # acceptance_ratio
            #if (np.log(u) <= r): # log_acceptance_ratio
                samples[:, s] = w_proposal
                w_current = w_proposal
            else:
                samples[:, s] = w_current
                copies[s] = 1

    return samples, copies


# Ex4, b)
sigma2 = 10 # 10
M = 10000 # Number of samples

wmh, copies = metropolis_hastings(t, X, sigma2, M)

# Remove duplicates
wmh = wmh[:, (copies == 0)]
# Remove burn-in samples
#wmh = wmh[:, 40:]

print("Effective #samples = " + str(wmh.shape[1]))

plt.figure()
plt.plot(range(wmh.shape[1]), wmh[0,:], 'b')
plt.xlabel('samples')
plt.ylabel('$w_1$')


# Make the plot of the samples
plt.figure()
plt.plot(wmh[0,0::10], wmh[1,0::10], 'b.')
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.title('Ex4, b)')


def unorm_log_posterior(w, t, X, sigma2):
    P = np.zeros((w.shape[1],1))

    for i in range(w.shape[1]):
        #P[i] = prior(w[:,i], sigma2) * likelihood(t, X, w[:,i])
        P[i] = log_prior(w[:, i], sigma2) + log_likelihood(t, X, w[:, i])

    return P

def unorm_posterior(w, t, X, sigma2):
    P = np.zeros((w.shape[1],1))

    for i in range(w.shape[1]):
        P[i] = prior(w[:,i], sigma2) * likelihood(t, X, w[:,i])
        #P[i] = log_prior(w[:, i], sigma2) + log_likelihood(t, X, w[:, i])

    return P


def posterior_contourplot(wsamples, t, X, sigma2):
    """Makes a contour plot of the probability function together with the data points (blue for t=0, and red for t=1).
        t - N-dim. vector of target values
        X - (Nx2) matrix containing the input vectors X for each data point
        w - parameters, either a (2x1) vector or a matrix of parameter vector samples
        prob - the function to plot contours for given by a function name to be called with prob(X, w) and must return a M-vector of probabilities
               (sigmoid as an example).
    """
    delta = 0.025
    x = np.arange(-2, 10, delta)
    y = np.arange(-2, 10, delta)
    XX, YY = np.meshgrid(x, y)

    #P = unorm_log_posterior(np.hstack([XX.reshape((np.prod(XX.shape), 1)), YY.reshape((np.prod(YY.shape), 1))]).T, t, X, sigma2).reshape(XX.shape)
    P = unorm_posterior(np.hstack([XX.reshape((np.prod(XX.shape), 1)), YY.reshape((np.prod(YY.shape), 1))]).T, t, X, sigma2).reshape(XX.shape)

    fig, ax = plt.subplots()
    CS = ax.contour(XX, YY, P)
    ax.clabel(CS, fontsize=9, inline=1)

    # Plot the data on top
    ax.plot(wsamples[0,0::10], wsamples[1, 0::10], 'b.')
    plt.xlabel('$w_1$')
    plt.ylabel('$w_2$')
    plt.title('log-Posterior and samples')


posterior_contourplot(wmh, t, X, sigma2)

# Ex4, c)
def predictive_prob(X, w):
    """Returns the probability of the class t=1, P(T_{new} = 1 | \vec{x}_{new}, \vec{t}, \vec{X}, \sigma^2). 
    Assumes 
    - X is a (NxD) matrix , 
    - w is a (DxM) vector of model parameters.
    returns a (Nx1) vector of the probabilities"""
    P = np.zeros((X.shape[0],1))
    for i in range(w.shape[1]):
        P += aux.sigmoid(X, w[:,i].reshape((2,1)))
        
    return P / w.shape[1]


aux.contourplot(t, X, wmh, predictive_prob)
plt.title('Ex4, (c)')

plt.show()
