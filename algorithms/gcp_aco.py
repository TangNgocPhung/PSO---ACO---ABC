"""Graph Coloring Problem – Ant Colony Optimization."""
import numpy as np
import random
import time


def count_conflicts(G, coloring):
    return sum(1 for u, v in G.edges() if coloring[u] == coloring[v])


def fitness(G, coloring, penalty=100):
    return penalty * count_conflicts(G, coloring) + len(set(coloring.values()))


def construct_coloring(G, pheromone, max_colors, alpha, beta):
    nodes = list(G.nodes())
    coloring = {}
    for v in nodes:
        neighbor_colors = {coloring[u] for u in G.neighbors(v) if u in coloring}
        heur = np.array([0.01 if c in neighbor_colors else 1.0 for c in range(max_colors)])
        tau = pheromone[v]
        p = (tau ** alpha) * (heur ** beta)
        s = p.sum()
        p = p / s if s > 0 else np.ones(max_colors) / max_colors
        c = int(np.random.choice(max_colors, p=p))
        coloring[v] = c
    return coloring


def aco_gcp(G, max_colors=5, num_ants=30, num_iter=80, alpha=1.0,
            beta=2.0, rho=0.3, Q=100.0, seed=42, progress_cb=None):
    random.seed(seed)
    np.random.seed(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    node_index = {v: i for i, v in enumerate(nodes)}
    pher = np.ones((n, max_colors))

    best_color, best_fit = None, float("inf")
    history = []
    start_t = time.time()

    for it in range(num_iter):
        all_solutions, all_fits = [], []
        for _ in range(num_ants):
            # reset pher index
            pher_arr = np.array([pher[node_index[v]] for v in nodes])
            coloring = {}
            for v in nodes:
                neighbor_colors = {coloring[u] for u in G.neighbors(v) if u in coloring}
                heur = np.array([0.01 if c in neighbor_colors else 1.0 for c in range(max_colors)])
                tau = pher[node_index[v]]
                p = (tau ** alpha) * (heur ** beta)
                s = p.sum()
                p = p / s if s > 0 else np.ones(max_colors) / max_colors
                coloring[v] = int(np.random.choice(max_colors, p=p))
            f = fitness(G, coloring)
            all_solutions.append(coloring)
            all_fits.append(f)
            if f < best_fit:
                best_fit, best_color = f, dict(coloring)

        pher *= (1 - rho)
        for sol, f in zip(all_solutions, all_fits):
            for v, c in sol.items():
                pher[node_index[v], c] += Q / (f + 1e-6)

        history.append(best_fit)
        if progress_cb:
            progress_cb(it + 1, num_iter, best_fit)

    runtime = time.time() - start_t
    conflicts = count_conflicts(G, best_color)
    return best_color, best_fit, conflicts, history, runtime
