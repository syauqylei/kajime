from abc import ABC
from typing import Any, Iterable, List, Union

import numpy as np

from src.solvers.solver import Solver
from src.utils import MatrixUtils

Feature = Union[np.ndarray, Iterable]


class Regression(ABC):
    _solver: Solver

    @property
    def solver(self):
        return self._solver


class LinearRegression(Regression):
    BETAS: List[float]
    XtX: np.ndarray
    Xty: np.ndarray

    def fit(self, feat: Feature, y: List[Any]):
        self.__calculate_betas(feat, y)

    def predict(self, x) -> float:
        prediction = 0
        for b in self.BETAS:
            prediction += b * x
        return prediction

    def __calculate_betas(self, feat: Feature, y: List[Any]):
        X = MatrixUtils.generate_x(feat)
        self.XtX = np.dot(X.T, X)
        self.Xty = np.dot(X.T, np.array(y))
        self.BETAS = np.dot(np.linalg.inv(self.XtX), self.Xty)
