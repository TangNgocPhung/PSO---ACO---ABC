"""VRPTW – Ant Colony Optimization (port từ notebook gốc).

Đặc điểm:
    - Mỗi con kiến xây dựng lời giải bằng cách chọn KHÁCH KHẢ THI (feasible)
      thoả mọi ràng buộc: time window, capacity, return-to-depot.
    - Khi không còn khách khả thi → đóng tuyến, sang xe mới.
    - Pheromone update: rank-based (top-3 kiến) + elitist (best toàn cục).
"""
import numpy as np
import random
import math
import time


# ---------------------------------------------------------------------------
#  SINH DATASET
# ---------------------------------------------------------------------------
def generate_solomon_like(n=20, capacity=200, seed=42):
    """Sinh dataset VRPTW kiểu Solomon (depot ở giữa + N khách)."""
    rng = np.random.default_rng(seed)
    customers = [{
        "id": 0, "x": 50.0, "y": 50.0, "demand": 0,
        "ready_time": 0, "due_date": 1000, "service_time": 0,
    }]
    for i in range(1, n + 1):
        x, y = rng.uniform(0, 100), rng.uniform(0, 100)
        demand = int(rng.integers(5, 30))
        ready = int(rng.integers(0, 400))
        due = ready + int(rng.integers(80, 250))
        service = int(rng.integers(5, 20))
        customers.append({
            "id": i, "x": x, "y": y, "demand": demand,
            "ready_time": ready, "due_date": due, "service_time": service,
        })
    return customers, capacity


# ---------------------------------------------------------------------------
#  HÀM PHỤ TRỢ
# ---------------------------------------------------------------------------
def distance_matrix(customers):
    n = len(customers)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = math.hypot(customers[i]["x"] - customers[j]["x"],
                                 customers[i]["y"] - customers[j]["y"])
    return D


def evaluate_routes(routes, customers, D, capacity,
                    lambda_veh=50, lambda_late=1000):
    """Tính fitness và các chỉ số chi tiết."""
    total_dist = 0.0
    total_tardy = 0.0
    cap_viol = 0
    for r in routes:
        prev = 0
        t = 0.0
        load = 0
        for cid in r:
            c = customers[cid]
            arrival = t + D[prev, cid]
            total_dist += D[prev, cid]
            start_service = max(arrival, c["ready_time"])
            if start_service > c["due_date"]:
                total_tardy += (start_service - c["due_date"])
            t = start_service + c["service_time"]
            load += c["demand"]
            prev = cid
        total_dist += D[prev, 0]
        if load > capacity:
            cap_viol += 1
    fitness = (total_dist
               + lambda_veh * len(routes)
               + lambda_late * total_tardy
               + 10000 * cap_viol)
    return fitness, total_dist, len(routes), total_tardy, cap_viol


# ---------------------------------------------------------------------------
#  CONSTRUCT SOLUTION (1 con kiến xây 1 lời giải)
# ---------------------------------------------------------------------------
def construct_solution_ant(pheromone, D, customers, capacity, alpha=1.0, beta=2.0):
    n = len(customers)
    unvisited = set(range(1, n))
    routes = []

    while unvisited:
        current_route = []
        cur = 0
        time_now = 0.0
        load = 0

        while True:
            # Tìm khách khả thi
            feasible = []
            for j in unvisited:
                c = customers[j]
                arrival = time_now + D[cur, j]
                start_service = max(arrival, c["ready_time"])
                if start_service > c["due_date"]:
                    continue   # vi phạm time window
                if load + c["demand"] > capacity:
                    continue   # vượt capacity
                back_to_depot = start_service + c["service_time"] + D[j, 0]
                if back_to_depot > customers[0]["due_date"]:
                    continue   # không kịp về depot
                feasible.append(j)

            if not feasible:
                break   # đóng tuyến, sang xe mới

            tau = pheromone[cur, feasible] ** alpha
            eta = (1.0 / (D[cur, feasible] + 1e-10)) ** beta
            probs = tau * eta
            s = probs.sum()
            probs = probs / s if s > 0 else np.ones(len(feasible)) / len(feasible)

            next_cust = int(np.random.choice(feasible, p=probs))

            c = customers[next_cust]
            arrival = time_now + D[cur, next_cust]
            start_service = max(arrival, c["ready_time"])
            time_now = start_service + c["service_time"]
            load += c["demand"]
            current_route.append(next_cust)
            unvisited.discard(next_cust)
            cur = next_cust

        if current_route:
            routes.append(current_route)
        else:
            # Khách không thể nào phục vụ trong khung – phục vụ ép buộc
            forced = unvisited.pop()
            routes.append([forced])

    return routes


# ---------------------------------------------------------------------------
#  ACO MAIN LOOP
# ---------------------------------------------------------------------------
def aco_vrptw(customers, capacity, num_ants=20, num_iter=100,
              alpha=1.0, beta=2.5, rho=0.15, Q=100.0,
              lambda_veh=50, lambda_late=1000,
              elitist=True, seed=42, progress_cb=None):
    """
    Trả về dict: best_routes, best_cost, history_best, history_avg,
                 runtime, pheromone, detail (total_dist, n_veh, tardy, cap_viol)
    """
    random.seed(seed)
    np.random.seed(seed)

    n = len(customers)
    D = distance_matrix(customers)

    pheromone = np.ones((n, n)) * 1.0
    best_routes = None
    best_cost = float("inf")
    history_best, history_avg = [], []

    start = time.time()

    for it in range(num_iter):
        ant_routes, ant_costs = [], []
        for k in range(num_ants):
            routes = construct_solution_ant(pheromone, D, customers, capacity, alpha, beta)
            cost, _, _, _, _ = evaluate_routes(routes, customers, D, capacity,
                                                lambda_veh, lambda_late)
            ant_routes.append(routes)
            ant_costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_routes = [r[:] for r in routes]

        # Bay hơi
        pheromone *= (1 - rho)

        # Bổ sung – rank-based top-3
        sorted_idx = np.argsort(ant_costs)
        for rank, idx in enumerate(sorted_idx[:3]):
            weight = (3 - rank) / 6.0  # 0.5, 0.33, 0.17
            for route in ant_routes[idx]:
                prev = 0
                for cid in route:
                    pheromone[prev, cid] += weight * Q / ant_costs[idx]
                    pheromone[cid, prev] += weight * Q / ant_costs[idx]
                    prev = cid
                pheromone[prev, 0] += weight * Q / ant_costs[idx]
                pheromone[0, prev] += weight * Q / ant_costs[idx]

        # Elitist: bổ sung mạnh cho best toàn cục
        if elitist and best_routes is not None:
            for route in best_routes:
                prev = 0
                for cid in route:
                    pheromone[prev, cid] += 2.0 * Q / best_cost
                    pheromone[cid, prev] += 2.0 * Q / best_cost
                    prev = cid
                pheromone[prev, 0] += 2.0 * Q / best_cost
                pheromone[0, prev] += 2.0 * Q / best_cost

        history_best.append(best_cost)
        history_avg.append(float(np.mean(ant_costs)))

        if progress_cb:
            progress_cb(it + 1, num_iter, best_cost)

    runtime = time.time() - start
    _, total_dist, n_veh, tardy, cap_v = evaluate_routes(
        best_routes, customers, D, capacity, lambda_veh, lambda_late)

    return {
        "best_routes": best_routes,
        "best_cost": best_cost,
        "history_best": history_best,
        "history_avg": history_avg,
        "runtime": runtime,
        "pheromone": pheromone,
        "total_dist": total_dist,
        "n_vehicles": n_veh,
        "tardiness": tardy,
        "cap_violations": cap_v,
        "distance_matrix": D,
    }
