import numpy as np
from constructive import ConstructiveNumber


def _mid(x):
    if isinstance(x, ConstructiveNumber):
        return x.get(0.5)
    return float(x)


def simulated_annealing(f, x0, T0=10.0, T_min=1e-5, alpha=0.95,
                        max_iter=5000, step=0.5):
    x = np.array(x0, dtype=float)
    fx = _mid(f(x))
    f_calls = 1
    best_x, best_f = x.copy(), fx
    history = [x.copy()]
    T = T0

    for _ in range(max_iter):
        if T < T_min:
            break
        candidate = x + np.random.uniform(-step, step, size=len(x))
        fc = _mid(f(candidate))
        f_calls += 1

        delta = fc - fx
        # принимаем ухудшение с вероятностью exp(-delta/T)
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            x, fx = candidate, fc
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        T *= alpha
        history.append(best_x.copy())

    return {"x": best_x, "f": best_f, "history": history,
            "f_calls": f_calls, "iters": len(history) - 1}


def differential_evolution(f, bounds, pop_size=20, max_iter=500, F=0.8, CR=0.9):
    n = len(bounds)
    f_calls = 0

    pop = np.array([[np.random.uniform(lo, hi) for lo, hi in bounds]
                    for _ in range(pop_size)])
    fitness = np.array([_mid(f(ind)) for ind in pop])
    f_calls += pop_size
    history = [pop[np.argmin(fitness)].copy()]

    for _ in range(max_iter):
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

            mutant = np.clip(a + F * (b - c),
                             [lo for lo, _ in bounds],
                             [hi for _, hi in bounds])
            cross = np.random.rand(n) < CR
            if not cross.any():
                cross[np.random.randint(n)] = True
            trial = np.where(cross, mutant, pop[i])

            ft = _mid(f(trial))
            f_calls += 1
            if ft < fitness[i]:
                pop[i], fitness[i] = trial, ft

        history.append(pop[np.argmin(fitness)].copy())

    best_idx = np.argmin(fitness)
    return {"x": pop[best_idx], "f": fitness[best_idx], "history": history,
            "f_calls": f_calls, "iters": len(history)}
