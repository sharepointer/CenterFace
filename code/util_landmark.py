import numpy as np


def log_landmark(v):
    if isinstance(v, tuple) or isinstance(v, list) or isinstance(v, np.ndarray):
        return [log_landmark(item) for item in v]

    base = np.exp(1)
    if abs(v) < base:
        return v / base

    if v > 0:
        return np.log(v)
    else:
        return -np.log(-v)


def exp_landmark(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return [exp_landmark(item) for item in v]
    elif isinstance(v, np.ndarray):
        return np.array([exp_landmark(item) for item in v], v.dtype)

    gate = 1
    base = np.exp(1)
    if abs(v) < gate:
        return v * base

    if v > 0:
        return np.exp(v)
    else:
        return -np.exp(-v)
