import numpy as np
import matplotlib.pyplot as plt


# Auxillary functions
def gauss(X, params):
    """params is a tuble of (mu, sigma)"""
    mu, sigma = params
    return np.exp(-(X-mu)**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))

def mean(X, params):
    return X
    
def var(X, params):
    mu = params[0]
    return (X-mu)**2
    
    
def skewness(X, params):
    mu, sigma = params
    return ((X-mu)/sigma)**3
    
def kurtosis(X, params):
    """params is a tuble of (mu, sigma)"""
    return ((X-mu)/sigma)**4 - 3
    
def entropy(X, params):
    """params is a tuble of (p, params2) where params2 are parameters to be passed to p
       and p must be callable,"""
    p = params[0]
    params2 = params[1:]
    return -np.log(p(X, params2))
    

def MonteCarloIntegration(X, func, params):
    """Given a sample set X, peform Monte Carlo integration to compute expectation of func applied to X."""
    N = X.shape[0]
    return np.sum(func(X, params)) / N
    

# Exercise 1, Gaussian Monte Carlo estimate
def GaussianMonteCarloIntegration(mu, sigma, Nlist):
    """Perform Monte Carlo integration using Gaussian distributed data"""
    mean_est = []
    kurtosis_est = []
    entropy_est = []
    for N in Nlist:
        # Here I reuse the same samples for the three expectations, but it is also ok to generate new ones for each function    
        X = np.random.default_rng().normal(mu, sigma, N)
        # mean_est.append(np.sum(mean(X)) / N)
        mean_est.append(MonteCarloIntegration(X, mean, None))
        # kurtosis_est.append(np.sum(kurtosis(X, (mu, sigma))) / N)
        kurtosis_est.append(MonteCarloIntegration(X, kurtosis, (mu,sigma)))
        # entropy_est.append(np.sum(entropy(X, gauss, (mu, sigma))) / N)
        entropy_est.append(MonteCarloIntegration(X, entropy, (gauss, mu,sigma)))
        
    return mean_est, kurtosis_est, entropy_est
    
    

mu = 2
sigma = 2 # sqrt(4)
Nlist = np.arange(10,5000, 10)
mean_est, kurtosis_est, entropy_est = GaussianMonteCarloIntegration(mu, sigma, Nlist)
plt.figure()
plt.plot(Nlist, mean_est)
plt.plot([0, Nlist[-1]],[mu, mu], 'r')
plt.xlabel('N')
plt.ylabel('E[X]')
plt.title('Ex.1 b)')

plt.figure()
plt.plot(Nlist, kurtosis_est)
plt.plot([0, Nlist[-1]],[0.0, 0.0], 'r')
plt.xlabel('N')
plt.ylabel('Kurt[X]')
plt.title('Ex.1 b)')

plt.figure()
plt.plot(Nlist, entropy_est)
ent = np.log(sigma*np.sqrt(2*np.pi*np.exp(1)))
plt.plot([0, Nlist[-1]],[ent, ent], 'r')
plt.xlabel('N')
plt.ylabel('H[X]')
plt.title('Ex.1 b)')


# Exercise 2, Rejection sampling based estimate

def targetdist(x, params = None):
    """The target distribution - an unnormalized PDF for the Laplace distribution with mu=2 and b = 1."""
    return np.exp(-np.abs(x-2))

#Ex.2 a)
x = np.arange(-20,20, 0.1)
mu2 = 2
sigma2 = 2
k = (sigma2*np.sqrt(2*np.pi))
plt.figure()
plt.plot(x, targetdist(x), 'b')
plt.plot(x, k*gauss(x, (mu2, sigma2)), 'r')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('(blue) target distribution, (red) k*gaussian distribution')
plt.title('Ex.2 a)')

plt.figure()
plt.plot(x, k*gauss(x, (mu2, sigma2)) / targetdist(x), 'b')
plt.xlabel('x')
plt.ylabel('(k*q(x)) / p(x)')
plt.title('Extra plot')


#Ex.2 b)
def rejectionsampler(mu, sigma, k, N):
    """Perform rejection sampling N times of the target distribution using a Gaussian proposal distribution.
    Returns M <= N samples that gets accepted."""
    
    X0 = np.random.default_rng().normal(mu, sigma, N)
    u0 = k * gauss(X0, (mu, sigma)) * np.random.default_rng().random(N)
    accepted = (u0 <= targetdist(X0))
    X = X0[accepted] # Select all accepted samples from the set of proposals.
    return X


# Make a histogram as a control
N = 1000
X = rejectionsampler(mu2, sigma2, k, N)
print('#Accepted samples = ' + str(X.shape[0]))
plt.figure()
plt.hist(X, 50) 
plt.title('Histogram of samples from the rejection sampler')



#Ex.2 c)
def RejectionMonteCarloIntegration(mu, sigma, k, Nlist):
    """Perform Monte Carlo integration using rejection sampling data"""
    mean_est = []
    var_est = []
    skewness_est = []
    kurtosis_est = []
    effectiveSamples = []
    for N in Nlist:
        # Here I reuse the same samples for the three expectations, but it is also ok to generate new ones for each function    
        X = rejectionsampler(mu, sigma, k, N)
        
        effectiveSamples.append(X.shape[0])
        mean_est.append(MonteCarloIntegration(X, mean, None))
        
        var_est.append(MonteCarloIntegration(X, var, (mean_est[-1], )))
        
        skewness_est.append(MonteCarloIntegration(X, skewness, (mean_est[-1], np.sqrt(var_est[-1]))))
        
        sample_mean = np.mean(X)
        sample_std = np.std(X)
        
        kurtosis_est.append(MonteCarloIntegration(X, kurtosis, (sample_mean, sample_std)))
        
        # Not possible to compute the entropy this way since target distribution is not normalized
        #entropy_est.append(MonteCarloIntegration(X, entropy, (targetdist, ))) 
        
    return mean_est, var_est, skewness_est, kurtosis_est, effectiveSamples
    

#Ex.2 d)
Nlist = np.arange(10,10000, 10)
mean_est, var_est, skewness_est, kurtosis_est, effectiveSamples = RejectionMonteCarloIntegration(mu2, sigma2, k, Nlist)
plt.figure()
plt.plot(effectiveSamples, mean_est, 'b.')
plt.plot([0, Nlist[-1]],[mu, mu], 'r')
plt.xlabel('N')
plt.ylabel('E[X]')
plt.title('Ex.2 d)')

plt.figure()
plt.plot(effectiveSamples, var_est, 'b.')
plt.plot([0, Nlist[-1]],[2.0, 2.0], 'r')
plt.xlabel('N')
plt.ylabel('Var[X]')
plt.title('Ex.2 d)')

plt.figure()
plt.plot(effectiveSamples, skewness_est, 'b.')
plt.plot([0, Nlist[-1]],[0.0, 0.0], 'r')
plt.xlabel('N')
plt.ylabel('Skew[X]')
plt.title('Ex.2 d)')

plt.figure()
plt.plot(effectiveSamples, kurtosis_est, 'b.')
plt.plot([0, Nlist[-1]],[3.0, 3.0], 'r')
plt.xlabel('N')
plt.ylabel('Kurt[X]')
plt.title('Extra plot')


plt.show()