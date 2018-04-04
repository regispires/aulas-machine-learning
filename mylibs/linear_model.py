import numpy as np

class MultivariateLinearRegression:
    
    def __init__(self, learning_rate, epoch):
        self.coefs = None
        self.learning_rate = learning_rate
        self.epoch = epoch 

    def predict_row(self, row):
        yhat = self.coefs[0]
        for i in range(len(row)):
            yhat += self.coefs[i + 1] * row[i]
        return yhat

    def predict(self, X):
        y_pred = np.zeros([X.shape[0]])
        for i in range(X.shape[1]):
            y_pred += self.coefs[i + 1] * X[:, i]
        return y_pred
        
    def fit(self, X, y):
        n_rows = X.shape[0]
        n_cols = X.shape[1]
        
        self.coefs = np.zeros([n_cols+1])
        for epoch in range(self.epoch):
            sum_error = 0
            for i in range(n_rows):
                yhat = self.predict_row(X[i])
                error = yhat - y[i]
                self.coefs[0] = self.coefs[0] - self.learning_rate * error
                for j in range(n_cols):
                    self.coefs[j + 1] = self.coefs[j + 1] - self.learning_rate * error * X[i, j]
            print(self.learning_rate, i, error, yhat)