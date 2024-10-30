import numpy as np


class MatrixUtils:

    @staticmethod
    def generate_X(x: np.ndarray):
        count = x.shape[0]
        bias_vector = np.ones((count, 1))
        X = np.reshape(x, (count, 1))
        X = np.append(bias_vector, X, axis=1)
        return X
