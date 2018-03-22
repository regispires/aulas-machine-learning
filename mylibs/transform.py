import numpy as np

def normalize(X):
    if len(X.shape) == 1:
        X[:] = (X - np.min(X)) / (np.max(X) - np.min(X))
        print('-len: {}, min: {}, max: {}'.format(len(X.shape), np.min(X), np.max(X)))
    else:
        n_cols = X.shape[1]
        for col in range(n_cols):
            X[:, col] = (X[:, col] - np.min(X[:, col])) / (np.max(X[:, col]) - np.min(X[:, col]))
