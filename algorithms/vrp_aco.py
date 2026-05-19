"""CVRP – Capacitated VRP với ACO."""
import numpy as np
import random
import time


def make_random_cvrp(n_customers=15, capacity=50, seed=42):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(n_customers + 1, 2))
    coords[0] = [50, 50]  # depot
    demands = rng.integers(5, 20, size=n_customers + 1)
    demands[0] = 0
    return coords, demands, capacity


def cvrp_distance(coords):
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = np.linalg.norm(coords[i] - coords[j])
    return d


def decode_routes(sequence, demands, capacity):
    """Chia chuỗi khách hàng thành các tour theo capacity."""
    routes = []
    cur, load = [], 0
    for c in sequence:
        if load + demands[c] > capacity:
            routes.append(cur)
            cur, load = [], 0
        cur.append(c)
        load += demands[c]
    if cur:
        routes.append(cur)
    return routes


def route_total_distance(routes, dist):
    total = 0
    for r in routes:
        prev = 0
        for c in r:
            total += dist[prev, c]
            prev = c
        total += dist[prev, 0]
    return total


def aco_cvrp(coords, demands, capacity, n_ants=20, n_iter=80,
             alpha=1.0, beta=2.0, rho=0.4, Q=100.0, seed=42, progress_cb=None):
    random.seed(seed)
    np.random.seed(seed)
    n = len(coords)
    dist = cvrp_distance(coords)
    eta = 1.0 / (dist + 1e-10)
    pher = np.ones((n, n))

    best_routes, best_len = None, float("inf")
    history = []
    start_t = time.time()

    for it in range(n_iter):
        for _ in range(n_ants):
            customers = list(range(1, n))
            random.shuffle(customers)
            # build chain by probabilistic selection from current
            sequence = []
            cur = 0
            remaining = set(customers)
            while remaining:
                cand = list(remaining)
                tau = pher[cur, cand]
                heur = eta[cur, cand]
                p = (tau ** alpha) * (heur ** beta)
                s = p.sum()
                p = p / s if s > 0 else np.ones(len(cand)) / len(cand)
                nxt = int(np.random.choice(cand, p=p))
                sequence.append(nxt)
                remaining.discard(nxt)
                cur = nxt
            routes = decode_routes(sequence, demands, capacity)
            length = route_total_distance(routes, dist)
            if length < best_len:
                best_len, best_routes = length, [r[:] for r in routes]

        pher *= (1 - rho)
        if best_routes:
            for r in best_routes:
                prev = 0
                for c in r:
                    pher[prev, c] += Q / best_len
                    pher[c, prev] += Q / best_len
                    prev = c
                pher[prev, 0] += Q / best_len
                pher[0, prev] += Q / best_len

        history.append(best_len)
        if progress_cb:
            progress_cb(it + 1, n_iter, best_len)

    runtime = time.time() - start_t
    return best_routes, best_len, history, runtime
