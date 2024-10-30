from typing import List
import numpy as np
from src.solvers.solver import Solver
from src.utils.matrix import MatrixUtils


class NormalEquation(Solver):
    def solve(self, y: np.ndarray, x: np.ndarray) -> List[float]:
        X = MatrixUtils.generate_X(x)
        betas = np.linalg.inv(X.T @ X) @ (X.T @ y)
        return [item for item in betas]
