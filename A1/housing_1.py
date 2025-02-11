import numpy
import matplotlib.pyplot as plt

# load data
train_data = numpy.loadtxt("boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set
train_mean = numpy.mean(t_train)
print("Mean of the training set is:", train_mean)

test_mean = numpy.mean(t_test)
print("Mean of the test set is:", test_mean)

# (b) RMSE function
# def rmse(t, tp):
#     ...
mean_array = numpy.zeros(X_train.shape[0])
def rmse(t, tp):
    N = X_train.shape[0]
    sum = 0.0
    for i in range (N):
        sum = sum + (t[i] - train_mean) ** 2
        mean_array[i] = train_mean
    rmse = numpy.sqrt(sum/N)
    return rmse

print(rmse(t_test, t_train))

# (c) visualization of results

plt.scatter(t_test, mean_array)

