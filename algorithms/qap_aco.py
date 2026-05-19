"""QAP – Quadratic Assignment Problem with ACO (8 departments × 8 locations)."""
import numpy as np
import random
import time

DEPARTMENT_NAMES = [
    "Tổng giám đốc", "Tài chính", "Nhân sự", "Kỹ thuật",
    "Sản xuất", "Marketing", "R&D", "Kho vận"
]
SHORT = ["TGĐ", "TC", "NS", "KT", "SX", "MKT", "R&D", "KV"]
LOCATION_NAMES = [f"P{i+1} (Tầng {i//2+1})" for i in range(8)]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

# Flow (số lần trao đổi/tuần giữa các phòng ban)
FLOW = np.array([
    [0, 10, 8, 4, 3, 7, 5, 2],
    [10, 0, 6, 2, 5, 4, 3, 1],
    [8, 6, 0, 5, 4, 3, 6, 4],
    [4, 2, 5, 0, 9, 2, 8, 6],
    [3, 5, 4, 9, 0, 3, 5, 9],
    [7, 4, 3, 2, 3, 0, 2, 4],
    [5, 3, 6, 8, 5, 2, 0, 3],
    [2, 1, 4, 6, 9, 4, 3, 0],
], dtype=float)

# Distance (m) giữa các phòng
DISTANCE = np.array([
    [0, 5, 10, 12, 8, 13, 18, 20],
    [5, 0, 5, 7, 10, 8, 13, 15],
    [10, 5, 0, 5, 12, 10, 8, 10],
    [12, 7, 5, 0, 10, 12, 10, 8],
    [8, 10, 12, 10, 0, 5, 10, 12],
    [13, 8, 10, 12, 5, 0, 5, 7],
    [18, 13, 8, 10, 10, 5, 0, 5],
    [20, 15, 10, 8, 12, 7, 5, 0],
], dtype=float)


def qap_cost(assignment, flow, distance):
    """assignment[i] = location của department i."""
    n = len(assignment)
    cost = 0.0
    for i in range(n):
        for j in range(n):
            cost += flow[i, j] * distance[assignment[i], assignment[j]]
    return cost


def aco_qap(flow, distance, num_ants=30, num_iter=150,
            alpha=1.0, beta=2.0, rho=0.15, Q=1000.0, seed=42, progress_cb=None):
    random.seed(seed)
    np.random.seed(seed)
    n = flow.shape[0]
    pher = np.ones((n, n))
    history = []
    best_assign, best_cost = None, float("inf")
    start_t = time.time()

    for it in range(num_iter):
        for _ in range(num_ants):
            assign = [-1] * n
            free_locs = list(range(n))
            for dept in range(n):
                tau = pher[dept, free_locs]
                # heuristic: ưu tiên location ít tải hơn
                heur = 1.0 / (distance[free_locs].sum(axis=1) + 1)
                p = (tau ** alpha) * (heur ** beta)
                s = p.sum()
                p = p / s if s > 0 else np.ones(len(free_locs)) / len(free_locs)
                chosen_idx = int(np.random.choice(len(free_locs), p=p))
                loc = free_locs.pop(chosen_idx)
                assign[dept] = loc
            cost = qap_cost(assign, flow, distance)
            if cost < best_cost:
                best_cost, best_assign = cost, assign[:]

        pher *= (1 - rho)
        if best_assign:
            for dept, loc in enumerate(best_assign):
                pher[dept, loc] += Q / best_cost

        history.append(best_cost)
        if progress_cb:
            progress_cb(it + 1, num_iter, best_cost)

    runtime = time.time() - start_t
    return best_assign, best_cost, history, runtime
