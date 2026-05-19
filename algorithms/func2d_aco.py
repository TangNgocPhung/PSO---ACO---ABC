"""Tối ưu hàm 2D (Rastrigin, Ackley, Sphere) bằng ACO liên tục."""
import numpy as np
import random
import time
import math


def rastrigin(x, y):
    return 20 + (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y))


def ackley(x, y):
    a, b, c = 20, 0.2, 2 * np.pi
    return (-a * np.exp(-b * np.sqrt(0.5 * (x ** 2 + y ** 2)))
            - np.exp(0.5 * (np.cos(c * x) + np.cos(c * y))) + a + np.e)


def sphere(x, y):
    return x ** 2 + y ** 2


def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


FUNCTIONS = {
    "Rastrigin": (rastrigin, (-5.12, 5.12), (0.0, 0.0, 0.0)),
    "Ackley": (ackley, (-5.0, 5.0), (0.0, 0.0, 0.0)),
    "Sphere": (sphere, (-5.0, 5.0), (0.0, 0.0, 0.0)),
    "Himmelblau": (himmelblau, (-5.0, 5.0), (3.0, 2.0, 0.0)),
}


def aco_function_opt(func, lo, hi, n_ants=40, n_iter=120, grid_m=30,
                     alpha=1.0, beta=2.0, rho=0.10, Q=10.0,
                     elitist=3, local_steps=15, top_frac=0.3,
                     seed=42, progress_cb=None):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Pheromone grid_m × grid_m
    grid_x = np.linspace(lo, hi, grid_m)
    grid_y = np.linspace(lo, hi, grid_m)
    pher = np.ones((grid_m, grid_m))
    # heuristic: 1/(f+eps) tại centroid
    heur = np.zeros((grid_m, grid_m))
    for i in range(grid_m):
        for j in range(grid_m):
            heur[i, j] = 1.0 / (func(grid_x[i], grid_y[j]) + 1.0)

    best_x, best_y, best_f = 0.0, 0.0, float("inf")
    history = []
    start_t = time.time()

    for it in range(n_iter):
        ant_results = []
        # Tính phân bố xác suất cell theo pher*heur
        weights = (pher ** alpha) * (heur ** beta)
        flat = weights.flatten()
        flat = flat / flat.sum()
        cells = rng.choice(grid_m * grid_m, size=n_ants, p=flat)

        for c in cells:
            i, j = c // grid_m, c % grid_m
            # sample uniformly trong cell
            dx = (hi - lo) / grid_m
            x = grid_x[i] + rng.uniform(-dx / 2, dx / 2)
            y = grid_y[j] + rng.uniform(-dx / 2, dx / 2)
            # local random walk
            for _ in range(local_steps):
                nx = x + rng.normal(0, dx / 4)
                ny = y + rng.normal(0, dx / 4)
                nx = np.clip(nx, lo, hi)
                ny = np.clip(ny, lo, hi)
                if func(nx, ny) < func(x, y):
                    x, y = nx, ny
            f = func(x, y)
            ant_results.append((x, y, f, i, j))
            if f < best_f:
                best_f, best_x, best_y = f, x, y

        # update pheromone (elitist)
        pher *= (1 - rho)
        ant_results.sort(key=lambda a: a[2])
        top_n = max(1, int(len(ant_results) * top_frac))
        for x, y, f, i, j in ant_results[:top_n]:
            pher[i, j] += Q / (f + 1.0)
        # elitist
        i_best = int((best_x - lo) / (hi - lo) * (grid_m - 1))
        j_best = int((best_y - lo) / (hi - lo) * (grid_m - 1))
        i_best = np.clip(i_best, 0, grid_m - 1)
        j_best = np.clip(j_best, 0, grid_m - 1)
        pher[i_best, j_best] += elitist * Q / (best_f + 1.0)

        history.append(best_f)
        if progress_cb:
            progress_cb(it + 1, n_iter, best_f)

    runtime = time.time() - start_t
    return best_x, best_y, best_f, history, pher, runtime
