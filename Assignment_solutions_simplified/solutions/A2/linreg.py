import numpy

class LinearRegression():
    """
    Linear regression implementation with
    regularization.
    """

    def __init__(self, lam=0, solver="solve"):
        """ Constructor for model.

        Parameters
        ----------
        lam : float
            The regularization parameter
        solver : str
            The solver that shall be used to approach 
            the linear systems of equations. 

            Two options so far: 
                - 'inverse': resorts to the matrix inverse 
                   of the data matrix
                - 'solve' : resorts to directly solving the
                   linear systems of equations (is usually
                   more precise.
        """
        
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

        # make sure that we have numpy arrays; also
        # reshape the array X to ensure that we have
        # a multidimensional numpy array (ndarray)
        X = numpy.array(X).reshape((X.shape[0], -1))
        t = numpy.array(t)

        # prepend a column of ones
        ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)           

        # compute weights
        diag = self.lam * len(X) * numpy.identity(X.shape[1])
        km = numpy.dot(X.T, X) + diag

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

        # make sure that we have numpy arrays; also
        # reshape the array X to ensure that we have
        # a multidimensional numpy array (ndarray)
        X = numpy.array(X).reshape((X.shape[0], -1))

        # prepend a column of ones
        ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)           

        # compute predictions
        predictions = numpy.dot(X, self.w)

        return predictions

