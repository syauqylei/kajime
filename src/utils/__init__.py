import numpy as np


def mse(y: np.ndarray, x: np.ndarray) -> np.floating:
    e_2 = (y - x) ** 2
    return np.sum(e_2) / len(e_2)
