

import numpy as np

## some multiclass methods that I never got to get working

def obj_multiclass(x0, X, y, alpha, n_class):
    n_samples, n_features = X.shape
    W = x0.reshape((n_features + 1, n_class-1))
    Wk = - W.sum(1)[:, None]
    W = np.concatenate((W, Wk), axis=1)
    X = np.concatenate((X, np.ones((n_samples, 1))), axis=1)
    Y = np.zeros((n_samples, n_class))
    Y[:] = - 1./(n_class - 1)
    for i in range(n_samples):
        Y[i, y[i]] = 1.

    L = np.abs(np.arange(n_class)[:, None] - np.arange(n_class))
    obj = (L[y] * np.fmax(X.dot(W) - Y, 0)).sum() / float(n_samples)

    Wt = W[:n_features]
    penalty = alpha * np.trace(Wt.T.dot(Wt))
    return obj + penalty


def multiclass_fit(X, y, alpha, n_class, maxiter=100000):
    """
    Multiclass classification with absolute error cost

    References
    ----------
    Lee, Yoonkyung, Yi Lin, and Grace Wahba. "Multicategory support
    vector machines: Theory and application to the classification of
    microarray data and satellite radiance data." Journal of the
    American Statistical Association 99.465 (2004): 67-81.
    """
    X = np.asarray(X)
    y = np.asarray(y) # XXX check its made of integers
    n_samples, n_features = X.shape

    x0 = np.random.randn((n_features + 1) * (n_class - 1))
    options = {'maxiter' : maxiter}
    sol = optimize.minimize(obj_multiclass, x0, jac=False,
                            args=(X, y, alpha, n_class), method='L-BFGS-B',
                            options=options)
    if not sol.success:
        print(sol.message)
    W = sol.x.reshape((n_features + 1, n_class-1))
    Wk = - W.sum(1)[:, None]
    W = np.concatenate((W, Wk), axis=1)
    return W

def multiclass_predict(X, W):
    n_samples, n_features = X.shape
    X = np.concatenate((X, np.ones((n_samples, 1))), axis=1)
    XW = X.dot(W)
    return np.argmax(XW, axis=1)

