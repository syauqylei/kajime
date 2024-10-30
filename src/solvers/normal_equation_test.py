import numpy as np

from src.solvers.normal_equation import NormalEquation
from src.utils import mse


def test_instatiate_NormalEquation(data_salary):
    data = data_salary.to_numpy()

    x = data[:, 0]
    y = data[:, 1]

    normalEq = NormalEquation()

    coefs = normalEq.solve(y, x)

    y_ = np.array([coefs[0] + coefs[1] * xi for xi in x])

    mse_ = mse(y, y_)
    assert mse_ < 32 * 1e6
