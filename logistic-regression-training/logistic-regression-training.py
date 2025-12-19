import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    w = np.zeros((X.shape[1],1))
    b = 0.0
    N = X.shape[0]

    for step in range(steps):
        a = (X @ w + b)
        z = _sigmoid(a)

        dz = z - y.reshape(-1,1)

        dw = (X.T @ dz) / N
        db =  np.mean(dz)

        w -= lr * dw
        b -= lr * db
    
    return (w.flatten(),b)
