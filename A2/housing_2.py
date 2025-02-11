import numpy
import pandas
import linweightreg
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
model_single = linweightreg.LinearRegression()
model_single.fit(X_train[:,0], t_train)
print("weights for a single variable: \n", model_single.w)
print("Husprisen starter på 23 tusinde dollars, når crimerate er 0, og falder med 0.4, når enheden for crimerate stiger med 1")

# (c) fit linear regression model using all features
model_multiple = linweightreg.LinearRegression()
model_multiple.fit(X_train, t_train)
print("weights for multiple variables: \n", model_multiple.w)
print("Husprisen starter på 31 tusiden dollars, når alle andre værdier er 0, og falder eller stiger, alt efter fortegnet på værdien, når de andre værdier stiger")

# (d) evaluation of results
model_single.predict(X_test[:,0])
model_multiple.predict(X_test)

def rmse(t, tp):
    N = t.shape[0]
    sum = 0.0
    for i in range (N):
        sum = sum + (t[i] - tp[i]) ** 2
    rmse = numpy.sqrt(sum/N)
    return rmse

print("RMSE for a single variable: ", rmse(t_test, model_single.p))
print("RMSE for multiple variables: ", rmse(t_test, model_multiple.p))

print("Plot of single variable")
# plt.scatter(model_single.p, t_test)
print("Plot of multiple variables")
plt.scatter(model_multiple.p, t_test)
plt.title("Rigtige vs forudsete værdier")
plt.ylabel("Rigtige værdier")
plt.xlabel("Forudsete værdier")
plt.show()
