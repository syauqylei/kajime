from types import FunctionType, LambdaType
import numpy as np

from typing import Dict, List

from numpy._core.multiarray import ndarray
from src.solvers.solver import Solver
from src.utils.linalg import Linalg
from src.utils.matrix import MatrixUtils


class GradientDescent(Solver):
    """A class to solve a regression problem using Gradient Descent method.

    bi  = b0 - grad L(b)/b

    ...

    Attributes
    ----------

    _eta: float
        Learning rate for gradient descent. it controls acceleration towards local minimum

    _stop_condition: float
        a condition where iteration of calculation stop


    Methods
    -------


    """

    _eta: float
    _count_iter: int

    def __init__(self, max_iter: int = 100, eta: float = 0.01, **kwargs):
        self._eta = eta
        self._max_iter = max_iter

    def _compute_gradient_cost(self, y: np.ndarray, x: np.ndarray, betas: np.ndarray) -> np.ndarray:
        X = MatrixUtils.generate_X(x)
        Y = np.reshape(y, (len(y), 1))
        grad = -2 * X.T @ (Y - self._linfunc(x, betas)) / Y.shape[0]
        return grad

    def solve(self, y, x) -> List[float]:
        """
        Return a coeficients of regression

            Parameters:
                y (numpy.ndarray): Target data
                x (numpy.ndarray): Feature data

            Return:
                A list of coeficients bo, b1 whre bo is intercept and b1 is gradient coeficients in linear equation
        """
        betas = np.random.rand(x.ndim + 1, 1)

        for i in range(self._max_iter):
            betas = betas - self._eta * self._compute_gradient_cost(y, x, betas)

        return betas[:,0]
