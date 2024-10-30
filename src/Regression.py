from abc import ABC
from typing import List, Literal

import numpy as np

from src.solvers.gradient_descent import GradientDescent
from src.solvers.normal_equation import NormalEquation
from src.solvers.solver import Solver
from src.utils.matrix import MatrixUtils

SolverType = Literal["GradientDescent", "NormalEquation"]


class Regression(ABC):
    _solver: Solver
    _coefs: List[float]
    _mse: np.floating
    _eta: np.floating
    _epoch: int


class LinearRegression(Regression):

    def __init__(
        self, eta: float = 0.01, epoch: int = 100, solver: SolverType = "NormalEquation"
    ):
        if solver == "NormalEquation":
            self._solver = NormalEquation()
        if solver == "GradientDescent":
            self._solver = GradientDescent(eta=eta, max_iter=epoch)

    def fit(self, feat: np.ndarray, y: np.ndarray):
        self._coefs = self._solver.solve(y, feat)

    def predict(self, x: np.ndarray) -> np.ndarray:

        X = MatrixUtils.generate_X(x)
        COEFS = np.array([[coef for coef in self._coefs] for _ in range(x.shape[0])])
        prediction = np.array([np.sum(item) for item in COEFS * X])

        return prediction
