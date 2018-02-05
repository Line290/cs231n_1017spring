import numpy as np

class KP:
    def __init__(self):
        self._x = None
        self._alpha = self._b = self._kernel = None
    
    # poly kernal function
    @staticmethod
    def _poly(x, y, p=4):
        return (x.dot(y.T) + 1) ** p
    
    # rbf kernal function
    @staticmethod
    def _rbf(x, y, gamma):
        return np.exp(-gamma * np.sum((x[..., None, :] - y) ** 2, axis=2))
        
    def fit(self, x, y, kernel="poly", p=None, gamma=None, lr=0.001, batch_size=128, epoch=10000):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        if kernel == "poly":
            p = 4 if p is None else p
            self._kernel = lambda x_, y_: self._poly(x_, y_, p)
        elif kernel == "rbf":
            gamma = 1 / x.shape[1] if gamma is None else gamma
            self._kernel = lambda x_, y_: self._rbf(x_, y_, gamma)
        else:
            raise NotImplementedError("Kernel '{}' has not defined".format(kernel))
        self._alpha = np.zeros(len(x))
        self._b = 0.
        self._x = x
        k_mat = self._kernel(x, x)
        for _ in range(epoch):
            indices = np.random.permutation(len(y))[:batch_size]
            k_mat_batch, y_batch = k_mat[indices], y[indices]
            err = -y_batch * (k_mat_batch.dot(self._alpha) + self._b)
            if np.max(err) < 0:
                continue
            mask = err >= 0
            delta = lr * y_batch[mask]
            self._alpha += np.sum(delta[..., None] * k_mat_batch[mask], axis=0)
            self._b += np.sum(delta)
    
    def predict(self, x, raw=False):
        x = np.atleast_2d(x).astype(np.float32)
        k_mat = self._kernel(self._x, x)
        y_pred = self._alpha.dot(k_mat) + self._b
        if raw:
            return y_pred
        return np.sign(y_pred).astype(np.float32)
    
class SVM(KP):        
    def fit(self, x, y, kernel="rbf", p=None, gamma=None, c=1, lr=0.0001, batch_size=128, epoch=10000):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        if kernel == "poly":
            p = 4 if p is None else p
            self._kernel = lambda x_, y_: self._poly(x_, y_, p)
        elif kernel == "rbf":
            gamma = 1 / x.shape[1] if gamma is None else gamma
            self._kernel = lambda x_, y_: self._rbf(x_, y_, gamma)
        else:
            raise NotImplementedError("Kernel '{}' has not defined".format(kernel))
        self._alpha = np.zeros(len(x))
        self._b = 0.
        self._x = x
        k_mat = self._kernel(x, x)
        k_mat_diag = np.diag(k_mat)
        for _ in range(epoch):
            self._alpha -= lr * (np.sum(self._alpha * k_mat, axis=1) + self._alpha * k_mat_diag) * 0.5
            indices = np.random.permutation(len(y))[:batch_size]
            k_mat_batch, y_batch = k_mat[indices], y[indices]
            err = 1 - y_batch * (k_mat_batch.dot(self._alpha) + self._b)
            if np.max(err) <= 0:
                continue
            mask = err > 0
            delta = c * lr * y_batch[mask]
            self._alpha += np.sum(delta[..., None] * k_mat_batch[mask], axis=0)
            self._b += np.sum(delta)
