import numpy

# NOTE: This template makes use of Python classes. If 
# you are not yet familiar with this concept, you can 
# find a short introduction here: 
# http://introtopython.org/classes.html

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
        newX = numpy.reshape(X, (X.shape[0], -1))
        newT = numpy.reshape(t, (X.shape[0], -1))
        newnewX = numpy.insert(newX, 0, 1, axis=1)
        XT = numpy.transpose(newnewX)
        XTX = numpy.dot(XT, newnewX)
        inverse = numpy.linalg.inv(XTX)
        XTXX = numpy.dot(inverse, XT)
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
        r = X.shape
        prediction = numpy.zeros((r[0], 1))
        print(r)
        for i in range(r[0]):
            for j in range(r[1]):
                prediction[i] = prediction[i] + X[i,j] * self.w[j]
                # det her skal nok bruge linReg
        self.p = prediction


        # TODO: YOUR CODE HERE
