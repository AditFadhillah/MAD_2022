import numpy
import pandas
import linreg
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

# (b) fit linear regression using only the first feature
model_single = linreg.LinearRegression()
model_single.fit(X_train[:,0], t_train)
print(model_single.w)

# (c) fit linear regression model using all features
model_several = linreg.LinearRegression()
model_several.fit(X_train, t_train)
print(model_several.w)

# (d) evaluation of results
model_single.predict(t_test)
model_several.predict(X_test)
print(model_single.p)
print(model_several.p)

mean_array_single = model_single.p
mean_array_several = model_several.p
def rmse(t, tp):
    N = X_train.shape[0]
    sum = 0.0
    for i in range (N):
        sum = sum + (t[i] - tp[i]) ** 2
    rmse = numpy.sqrt(sum/N)
    return rmse

print(rmse(t_test, mean_array_single))
print(rmse(t_test, mean_array_several))

