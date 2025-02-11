import numpy

class LinearRegression():
    """
    Linear regression implementation.
    """

    def __init__(self):
        
        pass
            
    def fit(self, X, t):
        """
        Fits the linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1]
        """
        X = numpy.reshape(X, (X.shape[0], -1))
        X = numpy.insert(X, 0, 1, axis=1)
        XTX = numpy.dot(X.T, X)
        inverse = numpy.linalg.inv(XTX)
        XTXX = numpy.dot(inverse, X.T)
        self.w = numpy.dot(XTXX, t)

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
        X = numpy.reshape(X, (X.shape[0], -1))
        X = numpy.insert(X, 0, 1, axis = 1)
        self.p = numpy.dot(X, self.w)
