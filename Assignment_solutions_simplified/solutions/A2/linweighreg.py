import numpy

# NOTE: This template makes use of Python classes. If 
# you are not yet familiar with this concept, you can 
# find a short introduction here: 
# http://introtopython.org/classes.html

class WeightedLinearRegression():
    """
    Weighted linear regression implementation.
    """

    def __init__(self, weights=None):
        
        self.weights = weights
            
    def fit(self, X, t):
        """
        Fits the weighted linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1]
        """        

        # make sure that we have Numpy arrays; also
        # reshape the target array to ensure that we have
        # a N-dimensional Numpy array (ndarray), see
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.ndarray.html
        X = numpy.array(X).reshape((len(X), -1))
        t = numpy.array(t).reshape((len(t), 1))

        # prepend a column of ones
        ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)

        if self.weights is None:
            self.weights = numpy.ones(len(X))
            
        A = numpy.identity(len(X), dtype=numpy.float64)
        for i in range(len(self.weights)):
            A[i,i] = self.weights[i]

        
        # compute weights (solve system)
        a = numpy.dot(X.T, A)
        a = numpy.dot(a, X)
        b = numpy.dot(X.T, A)
        b = numpy.dot(b, t)

        self.w = numpy.linalg.solve(a,b)    
                
    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """                     

        # make sure that we have Numpy arrays; also
        # reshape the target array to ensure that we have
        # a N-dimensional Numpy array (ndarray), see
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.ndarray.html
        X = numpy.array(X).reshape((len(X), -1))

        # prepend a column of ones
        ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)           

        # compute predictions
        predictions = numpy.dot(X, self.w)

        return predictions

