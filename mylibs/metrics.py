import numpy as np

def mse(y, y_pred):
    n = len(y)
    return np.sum((y - y_pred) ** 2) / n
