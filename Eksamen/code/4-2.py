# %%
import csv
import numpy as np
from heapq import nsmallest
import matplotlib.pyplot as plt

# %%
train_data = np.loadtxt("../data/galaxies_train.csv", delimiter=",", skiprows=1)
test_data = np.loadtxt("../data/galaxies_test.csv", delimiter=",", skiprows=1)

# %%
X_train = train_data[:,1:]
t_train = train_data[:,0]
X_test = test_data[:,1:]
t_test = test_data[:,0]
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of attributes: %i" % X_train.shape[1])

# %%
# NOTE: You are supposed to use this strucuture, i.e., 
# the pre-defined functions and variables. If you 
# have difficulties to keep this structure, you ARE 
# ALLOWED to adapt/change the code structure slightly!
# You might also want to add additional functions or
# variables.

class NearestNeighborRegressor:
    
    def __init__(self, n_neighbors=3, dist_measure="euclidean", dist_matrix=None):
        """
        Initializes the model.
        
        Parameters
        ----------
        n_neighbors : The number of nearest neigbhors (default 1)
        dist_measure : The distance measure used (default "euclidean")
        dist_matrix : The distance matrix if needed (default "None")
        """
        
        self.n_neighbors = n_neighbors
        self.dist_measure = dist_measure
        self.dist_matrix = dist_matrix
        self.lam = 1.0
    
    def fit(self, X, t):
        """
        Fits the nearest neighbor regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of length n_samples
        """ 
        
        self.X_train = X
        self.t_train = t
    
    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of length n_samples
        """         
        
        predictions = []
        if self.dist_measure == "euclidean":
            for i in range(X.shape[0]):
                distance = []
                for j in range(X_train.shape[0]):
                    distance.append(np.linalg.norm(X_train[j] - X[i]))
                smallestN = nsmallest(self.n_neighbors, distance)
                tempFeatures = np.zeros((self.n_neighbors, X.shape[1]))
                tempLabels = []
                for j in range(self.n_neighbors):
                    tempFeatures[j] = self.X_train[distance.index(smallestN[j])]
                    tempLabels.append(self.t_train[distance.index(smallestN[j])])
                tempLabels = np.array(tempLabels)
                predictionWeights, lort1, lort2, lort3 = np.linalg.lstsq(tempFeatures.T.dot(tempFeatures) + self.lam * np.identity(X.shape[1]), tempFeatures.T.dot(tempLabels))
                # print(predictionWeights)
                predictions.append(predictionWeights.dot(X[i]))
            predictions = np.array(predictions)
            print("predictions: ", predictions)
            return predictions

# %%
KNN = NearestNeighborRegressor(n_neighbors = 15)
KNN.fit(X_train, t_train)
prediction = KNN.predict(X_test)
print("prediction: ", prediction)
def rmse(t, tp):
        N = t.shape[0]
        sum = 0.0
        for i in range (N):
            sum = sum + (t[i] - tp[i]) ** 2
        rmse = np.sqrt(sum/N)
        return rmse
rmseTest = rmse(prediction, t_test)
print("RMSE: ", rmseTest)
plt.scatter(t_test, prediction)
plt.title("True redshift vs predicted redshift")
plt.ylabel("Real values")
plt.xlabel("Predicted values")
plt.xlim([0,7])
plt.ylim([0,7])
plt.show()

# %%



