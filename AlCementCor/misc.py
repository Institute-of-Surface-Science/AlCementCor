import numpy as np


def is_diverged(value):
    return value > 1 or np.isnan(value)


def check_divergence(value):
    if is_diverged(value):
        raise ValueError("ERROR: Calculation diverged!")
