import numpy as np

from pandas import DataFrame
from src.solvers.gradient_descent import GradientDescent


def test_instatiate_class_GradientDescent():

    gdSolver = GradientDescent()

    assert gdSolver._eta == 0.01
    assert gdSolver._max_iter == 100


def test_run_GradientDescent(data_salary: DataFrame):
    np_arr = data_salary.to_numpy()

    # pre-process data to x , y

    x = np_arr[:, 0]
    y = np_arr[:, 1]

    gdSolver = GradientDescent(eta=0.01,max_iter=2500)

    coefs = gdSolver.solve(y, x)

    assert coefs[0] == 26780.09915063
    assert coefs[1] == 9312.57512673
