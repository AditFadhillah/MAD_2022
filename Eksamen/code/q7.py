import numpy as np
import statistics
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def loaddata(filename):
    """Load the balloon data set from filename and return t, X
        t - N-dim. vector of target (temperature) values
        X - N-dim. vector containing the inputs (lift) x for each data point
    """
    # Load data set from CSV file
    Xt = np.loadtxt(filename, delimiter=',')

    # Split into data matrix and target vector
    X = Xt[:,:-1]
    t = Xt[:,-1]
    
    return t, X

# Load data
t, X = loaddata('../data/seedsDataset.txt')

def normalize(samples):
    means = samples.mean(axis = 0)
    deviations = []
    norm = np.zeros(samples.shape)

    for c in samples.T:
        sampleStandardDeviation = statistics.stdev(c)
        deviations.append(sampleStandardDeviation)

    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            norm[i, j] = (samples[i, j] - means[j]) / deviations[j]
    return norm

# I use this, so I can do d, e
normalizedData = normalize(X)
KMean = KMeans(n_clusters = 3, random_state = 0)
KMean.fit_predict(normalizedData)
centroids = KMean.cluster_centers_

# from my assignement 5
def __PCA(data):
    meanTrainingFeatures = np.mean(data.T, axis = 1)

    data_cent = data.T - meanTrainingFeatures.reshape((-1, 1))
    data_cent = np.cov(data_cent)
    PCevals, PCevecs = np.linalg.eigh(data_cent)
    PCevals = np.flip(PCevals, 0)
    PCevecs = np.flip(PCevecs, 1)
    return PCevals, PCevecs

normEvals, normEvecs = __PCA(normalizedData)
cenEvals, cenEvecs = __PCA(centroids)

def __transformData(features, PCevecs):
    return np.dot(features,  PCevecs[:, 0:2])

normTransformed = __transformData(normalizedData, normEvecs)
cenTransformed = __transformData(centroids, cenEvecs)


def __visualizeLabels(features, referenceLabels, centroids):
    plt.figure()
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    y = referenceLabels

    plt.scatter(features[:, 0], features[:, 1], c = y, cmap = cmap_bold, s = 10)
    # added scatter for centroids.
    plt.scatter(centroids[:, 0], centroids[:, 1], color="black", s = 100)
    plt.show()

__visualizeLabels(normTransformed, t, cenTransformed)