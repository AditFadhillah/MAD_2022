import matplotlib.pyplot as plt
import numpy as np
import linreg

solver = "solve"

def lossFunction(X, t, lam, verbose=0):

    loss = 0

    for i in range(len(X)):
        X_train = np.delete(X, i, 0)
        t_train = np.delete(t, i, 0)

        model = linreg.LinearRegression(lam=lam, solver=solver)
        model.fit(X_train, t_train)

        X_val = X[i].reshape((1, X_train.shape[1]))
        t_val = t[i].reshape((1, 1))
        model.predict(X_val)

        loss += (t_val[0,0] - model.p[0,0]) ** 2.0

    loss = loss / len(X)

    if verbose > 0:
        print("lam=%.10f and loss=%.10f" % (lam, loss))

    return loss

raw = np.genfromtxt('men-olympics-100.txt', delimiter=' ')

t = raw[:,1].reshape((len(raw),1))

lambdaValues = np.logspace(-8, 0, 100, base=10)
print("1st Order Polynomial: ")
X = raw[:,0].reshape((len(raw),1))

resultA = np.array([lossFunction(X, t, lam) for lam in lambdaValues])
bestLambdaA = lambdaValues[np.argmin(resultA)]
print("Best lambda value: %.10f" % bestLambdaA)

modelZero = linreg.LinearRegression(lam=0.0, solver=solver)
modelZero.fit(X, t)
print("Optimal coefficients for lam=%.10f: \n%s" % (0.0, str(modelZero.w)))

modelBest = linreg.LinearRegression(lam=bestLambdaA, solver=solver)
modelBest.fit(X, t)
print("Optimal coefficients for lam=%.10f: \n%s" % (bestLambdaA, str(modelBest.w)))

plt.figure()
plt.plot(lambdaValues, resultA, "bo", markersize=3)
plt.title("Funktion af $\lambda$ på 1st-degree polynomialt fit")
plt.ylabel("Error")
plt.xlabel("$\lambda$ (logscale)")
plt.xscale("log")
plt.show()

print("4th Degree Polynomial")
X4 = np.empty((len(raw[:,0]),4))
X4[:,0] = raw[:,0]
X4[:,1] = raw[:,0] ** 2
X4[:,2] = raw[:,0] ** 3
X4[:,3] = raw[:,0] ** 4

resultB = np.array([lossFunction(X4, t, lam) for lam in lambdaValues])
bestLambdab = lambdaValues[np.argmin(resultB)]
print("Best lambda value: %.10f" % bestLambdab)

modelZero = linreg.LinearRegression(lam=0.0, solver=solver)
modelZero.fit(X4, t)
print("Optimal coefficients for lam=%.10f: \n%s" % (0.0, str(modelZero.w)))

modelBest = linreg.LinearRegression(lam=bestLambdab, solver=solver)
modelBest.fit(X4, t)
print("Optimal coefficients for lam=%.10f: \n%s" % (bestLambdab, str(modelBest.w)))

plt.figure()
plt.plot(lambdaValues, resultB, "bo", markersize=3)
plt.title("Funktion af $\lambda$ på 4th-degree polynomialt fit")
plt.ylabel("Error")
plt.xlabel("$\lambda$ (logscale)" )
plt.xscale("log")
plt.show()
