import os
import pytest
import pandas as pd


@pytest.fixture(scope="session")
def data_salary():
    cwd = os.getcwd()
    data = pd.read_csv(f"{cwd}/datasets/Salary_Data.csv")
    return data
