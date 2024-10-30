import numpy as np


class Linalg:

    @staticmethod
    def L2(y: np.ndarray, fx: np.ndarray):
        sum = np.average(np.abs(y - fx))
        return sum
