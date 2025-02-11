import numpy
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
print("Model coefficients using only the first feature:")
print(model_single.w)

# (c) fit linear regression model using all features
model_all = linreg.LinearRegression()
model_all.fit(X_train, t_train)
print("Model coefficients using all features:")
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(precision=2)
print(model_all.w)

# (d) evaluation of results
def rmse(t, tp):
    t = t.reshape((len(t), 1))
    tp = tp.reshape((len(tp), 1))
    return numpy.sqrt(numpy.mean((t - tp)**2))

# single feature
preds_single = model_single.predict(X_test[:,0])
print("Single Feature RMSE: %f" % rmse(preds_single, t_test))
plt.figure()
plt.scatter(t_test, preds_single)
plt.xlabel("House Prices")
plt.ylabel("Predicted House Prices")
plt.xlim([0,50])
plt.ylim([0,50])
plt.title("Single Feature (RMSE=%f)" % rmse(preds_single, t_test))

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
