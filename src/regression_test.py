import pytest

from src.regression import LinearRegression
from src.solvers.gradient_descent import GradientDescent
from src.solvers.normal_equation import NormalEquation
from src.utils import mse


@pytest.fixture(scope="module")
def data(data_salary):
    return data_salary.to_numpy()


def test_instatiate_LinearRegression(data):
    linreg = LinearRegression()

    assert isinstance(linreg._solver, NormalEquation)

    x = data[:, 0]
    y = data[:, 1]

    linreg.fit(x, y)

    y_pred = linreg.predict(x)

    error = mse(y, y_pred)

    assert error < 32e6

    linreg = LinearRegression(solver="GradientDescent")

    assert isinstance(linreg._solver, GradientDescent)

    linreg.fit(x, y)

    y_pred = linreg.predict(x)

    error = mse(y, y_pred)

    assert error < 85e6
    assert error > 35e6
