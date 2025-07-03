import numpy as np
import cvxpy as cp
from concurrent.futures import ProcessPoolExecutor


class RLFit:
    def __init__(self):
        raise NotImplementedError

    def fit(self, X, Y):
        raise NotImplementedError
