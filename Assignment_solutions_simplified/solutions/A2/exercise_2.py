import matplotlib.pyplot as plt
import numpy as np
import linreg

solver = "solve"

def loocv(X, t, lam, verbose=0):

    loss = 0

    for i in range(len(X)):

        # "remove" the i-th data point; the 
        # np.delete function returns a "copy"
        # of the array X/t that contains all
        # but the i-th data point
        X_train = np.delete(X, i, 0)
        t_train = np.delete(t, i, 0)

        # fit model on training data
        model = linreg.LinearRegression(lam=lam, solver=solver)
        model.fit(X_train, t_train)

        # get single prediction
        X_val = X[i].reshape((1, len(X[i])))
        t_val = t[i].reshape((1, 1))
        pred = model.predict(X_val)

        # compute squared loss and add
        # the result to the overall loss
        loss += (t_val - pred) ** 2.0

    loss = loss / len(X)

    if verbose > 0:
        print("lam=%.10f and loss=%.10f" % (lam, loss))

    return loss[0,0]

# import data
raw = np.genfromtxt('men-olympics-100.txt', delimiter=' ')

# target vector
t = raw[:,1].reshape((len(raw),1))

# lambda grid
lams = np.logspace(-8, 0, 100, base=10)

# (a) first order polynomial
print("--------------------")
print("1st Order Polynomial")
print("--------------------")
X = raw[:,0].reshape((len(raw),1))

result_a = np.array([loocv(X, t, lam) for lam in lams])
best_lam_a = lams[np.argmin(result_a)]
print("Best lambda value: %.10f" % best_lam_a)

model_zero = linreg.LinearRegression(lam=0.0, solver=solver)
model_zero.fit(X, t)
print("Optimal coefficients for lam=%.10f: %s" % (0.0, str(model_zero.w)))

model_best = linreg.LinearRegression(lam=best_lam_a, solver=solver)
model_best.fit(X, t)
print("Optimal coefficients for lam=%.10f: %s" % (best_lam_a, str(model_best.w)))

plt.figure()
plt.plot(lams, result_a, "bo", markersize=3)
plt.title("LOOCVE as a function of $\lambda$ on 1st-degree polynonial fit")
plt.ylabel("Error")
plt.xlabel("$\lambda$ (logscale)" )
plt.xscale("log")
plt.show()

# (b) fourth order polynomial
print("---------------------")
print("4th Degree Polynomial")
print("---------------------")
X_4 = np.empty((len(raw[:,0]),4))
X_4[:,0] = raw[:,0]
X_4[:,1] = raw[:,0]**2
X_4[:,2] = raw[:,0]**3
X_4[:,3] = raw[:,0]**4

result_b = np.array([loocv(X_4, t, lam) for lam in lams])
best_lam_b = lams[np.argmin(result_b)]
print("Best lambda value: %.10f" % best_lam_b)

model_zero = linreg.LinearRegression(lam=0.0, solver=solver)
model_zero.fit(X_4, t)
print("Optimal coefficients for lam=%.10f: %s" % (0.0, str(model_zero.w)))

model_best = linreg.LinearRegression(lam=best_lam_b, solver=solver)
model_best.fit(X_4, t)
print("Optimal coefficients for lam=%.10f: %s" % (best_lam_b, str(model_best.w)))

plt.figure()
plt.plot(lams, result_b, "bo", markersize=3)
plt.title("LOOCVE as a function of $\lambda$ on 4th-degree polynonial fit")
plt.ylabel("Error")
plt.xlabel("$\lambda$ (logscale)" )
plt.xscale("log")
plt.show()

