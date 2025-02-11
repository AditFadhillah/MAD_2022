import numpy

class LinearRegression():
    """
    Linear regression implementation with
    regularization.
    """

    def __init__(self, lam=0, solver="inverse"):        
        self.lam = lam
        self.solver = solver

        assert self.solver in ["inverse", "solve"]
            
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
        t = numpy.reshape(t, (len(t),1))

        diagonal = self.lam * len(X) * numpy.identity(X.shape[1])
        km = numpy.dot(X.T, X) + diagonal

        if self.solver == "solve":
            self.w = numpy.linalg.solve(km, numpy.dot(X.T, t))

        elif self.solver == "inverse":
            self.w = numpy.linalg.inv(km)
            self.w = numpy.dot(self.w, X.T)
            self.w = numpy.dot(self.w, t)

        else:
            raise Exception("Unknown solver!")
                
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
