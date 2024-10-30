import numpy as np

from abc import ABC, abstractmethod
from typing import List, TypedDict

from src.utils.matrix import MatrixUtils


class MetadataSolver(TypedDict):
    iteration: List[int]
    coefs: List[np.ndarray]
    loss: List[float]


class Solver(ABC):
    _metadata: MetadataSolver
    _betas: np.ndarray
    L2: np.floating

    def _linfunc(self, x: np.ndarray, betas: np.ndarray) -> np.ndarray:
        X = MatrixUtils.generate_X(x)
        y = np.dot(X, betas)
        return y

    def _compute_cost(self, y: np.ndarray, x: np.ndarray, betas: np.ndarray):
        return np.average((np.abs(y - self._linfunc(x, betas))) ** 2)

    @abstractmethod
    def solve(self, y: np.ndarray, x: np.ndarray) -> List[float]: ...
