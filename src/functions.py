import numpy as np
from constructive import ConstructiveNumber


def _cn(x, eps=1e-10):
    if isinstance(x, ConstructiveNumber):
        return x
    return ConstructiveNumber.from_real(float(x), eps)


# f(x) = 10n + sum(xi^2 - 10*cos(2pi*xi)), много локальных минимумов
def rastrigin(x):
    n = len(x)
    result = _cn(10 * n)
    for xi in x:
        xi_cn = _cn(xi)
        cos_val = _cn(np.cos(2 * np.pi * float(xi_cn)))
        result = result + xi_cn * xi_cn - cos_val * 10
    return result


def ackley(x):
    assert len(x) == 2
    x0, x1 = float(_cn(x[0])), float(_cn(x[1]))
    val = (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x0**2 + x1**2)))
           - np.exp(0.5 * (np.cos(2 * np.pi * x0) + np.cos(2 * np.pi * x1)))
           + np.e + 20)
    return _cn(val)


# аналог функции из десмоса: sin(x)*cos(y) + z^2
def desmos_func(x):
    assert len(x) == 3
    x0, x1, x2 = float(_cn(x[0])), float(_cn(x[1])), float(_cn(x[2]))
    return _cn(np.sin(x0) * np.cos(x1) + x2**2)
