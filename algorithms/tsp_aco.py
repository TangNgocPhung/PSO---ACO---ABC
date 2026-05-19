"""TSP – Ant Colony Optimization."""
import numpy as np
import random
import time


def euclid_matrix(coords):
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = np.linalg.norm(coords[i] - coords[j])
    return d


def aco_tsp(coords, n_ants=30, n_iter=100, alpha=1.0, beta=2.0,
            rho=0.5, Q=100.0, seed=42, progress_cb=None):
    rng = random.Random(seed)
    np.random.seed(seed)

    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    dist = euclid_matrix(coords)
    eta = 1.0 / (dist + 1e-10)
    pher = np.ones((n, n))

    best_path, best_len = None, float("inf")
    history = []

    start = time.time()
    for it in range(n_iter):
        all_paths, all_lens = [], []
        for _ in range(n_ants):
            visited = [rng.randrange(n)]
            while len(visited) < n:
                cur = visited[-1]
                cand = [j for j in range(n) if j not in visited]
                probs = (pher[cur, cand] ** alpha) * (eta[cur, cand] ** beta)
                s = probs.sum()
                if s == 0:
                    nxt = rng.choice(cand)
                else:
                    probs = probs / s
                    nxt = np.random.choice(cand, p=probs)
                visited.append(int(nxt))
            length = sum(dist[visited[i], visited[i + 1]] for i in range(n - 1))
            length += dist[visited[-1], visited[0]]
            all_paths.append(visited)
            all_lens.append(length)
            if length < best_len:
                best_len = length
                best_path = visited[:]

        # Bay hơi
        pher *= (1 - rho)
        # Bổ sung pheromone
        for path, length in zip(all_paths, all_lens):
            deposit = Q / length
            for i in range(n - 1):
                pher[path[i], path[i + 1]] += deposit
                pher[path[i + 1], path[i]] += deposit
            pher[path[-1], path[0]] += deposit
            pher[path[0], path[-1]] += deposit

        history.append(best_len)
        if progress_cb:
            progress_cb(it + 1, n_iter, best_len)

    runtime = time.time() - start
    return best_path, best_len, history, runtime


# Tập hợp dữ liệu mẫu (TSPLIB-like)
SAMPLE_DATASETS = {
    "berlin52 (52 thành phố)": np.array([
        [565, 575], [25, 185], [345, 750], [945, 685], [845, 655], [880, 660],
        [25, 230], [525, 1000], [580, 1175], [650, 1130], [1605, 620], [1220, 580],
        [1465, 200], [1530, 5], [845, 680], [725, 370], [145, 665], [415, 635],
        [510, 875], [560, 365], [300, 465], [520, 585], [480, 415], [835, 625],
        [975, 580], [1215, 245], [1320, 315], [1250, 400], [660, 180], [410, 250],
        [420, 555], [575, 665], [1150, 1160], [700, 580], [685, 595], [685, 610],
        [770, 610], [795, 645], [720, 635], [760, 650], [475, 960], [95, 260],
        [875, 920], [700, 500], [555, 815], [830, 485], [1170, 65], [830, 610],
        [605, 625], [595, 360], [1340, 725], [1740, 245]
    ], dtype=float),
    "Tam giác đều (3)": np.array([[0, 0], [1, 0], [0.5, 0.866]], dtype=float),
    "Lưới 10x1 (10)": np.array([[i, 0] for i in range(10)], dtype=float),
    "Ngẫu nhiên (20)": None,
    "Ngẫu nhiên (50)": None,
}


def random_cities(n, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 100, size=(n, 2))
