import numpy
import linweighreg
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

# (a) fit linear regression model using all features
model_all = linweighreg.WeightedLinearRegression(weights=t_train**2)
model_all.fit(X_train, t_train)

# (c) evaluation of results
def rmse(t, tp):
    t = t.reshape((len(t), 1))
    tp = tp.reshape((len(tp), 1))
    return numpy.sqrt(numpy.mean((t - tp)**2))

# all features
preds_all = model_all.predict(X_test)
print("RMSE for all features: %f" % rmse(preds_all, t_test))
plt.figure()
plt.scatter(t_test, preds_all)
plt.xlabel("House Prices")
plt.ylabel("Predicted House Prices")
plt.xlim([0,50])
plt.ylim([0,50])
plt.title("All Features (RMSE=%f)" % rmse(preds_all, t_test))
plt.show()
