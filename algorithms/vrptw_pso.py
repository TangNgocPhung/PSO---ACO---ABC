"""VRPTW – Vehicle Routing Problem with Time Windows (PSO + swap operator)."""
import numpy as np
import random
import math
import time


def make_solomon_like(n=20, capacity=200, seed=42):
    """Tạo dữ liệu kiểu Solomon (depot + khách)."""
    rng = np.random.default_rng(seed)
    customers = []
    # depot index 0
    customers.append({
        "id": 0, "x": 50.0, "y": 50.0, "demand": 0,
        "ready_time": 0, "due_date": 1000, "service_time": 0,
    })
    for i in range(1, n + 1):
        x, y = rng.uniform(0, 100), rng.uniform(0, 100)
        demand = int(rng.integers(5, 30))
        ready = int(rng.integers(0, 300))
        due = ready + int(rng.integers(60, 300))
        service = int(rng.integers(5, 20))
        customers.append({
            "id": i, "x": x, "y": y, "demand": demand,
            "ready_time": ready, "due_date": due, "service_time": service,
        })
    return customers, capacity


def build_distance_matrix(customers):
    n = len(customers)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = math.hypot(customers[i]["x"] - customers[j]["x"],
                                 customers[i]["y"] - customers[j]["y"])
    return d


def decode_particle_to_routes(particle, depot, capacity):
    routes = []
    cur, load = [], 0
    for c in particle:
        if load + c["demand"] > capacity:
            routes.append(cur)
            cur, load = [], 0
        cur.append(c)
        load += c["demand"]
    if cur:
        routes.append(cur)
    return routes


def evaluate_fitness(particle, depot, capacity, dist):
    routes = decode_particle_to_routes(particle, depot, capacity)
    total_dist = 0.0
    penalty = 0.0
    for r in routes:
        prev_id = depot["id"]
        time_now = 0.0
        for c in r:
            d = dist[prev_id, c["id"]]
            total_dist += d
            time_now += d
            if time_now < c["ready_time"]:
                time_now = c["ready_time"]
            if time_now > c["due_date"]:
                penalty += (time_now - c["due_date"]) * 100
            time_now += c["service_time"]
            prev_id = c["id"]
        total_dist += dist[prev_id, depot["id"]]
    vehicle_penalty = len(routes) * 50
    return total_dist + penalty + vehicle_penalty, total_dist, len(routes)


def swap_operator(particle):
    new = particle[:]
    i, j = random.sample(range(len(new)), 2)
    new[i], new[j] = new[j], new[i]
    return new


def pso_vrptw(customers, capacity, n_particles=30, n_iter=80, seed=42, progress_cb=None):
    random.seed(seed)
    np.random.seed(seed)
    depot = customers[0]
    others = customers[1:]
    dist = build_distance_matrix(customers)

    particles = []
    for _ in range(n_particles):
        p = others[:]
        random.shuffle(p)
        particles.append(p)

    pbest = [p[:] for p in particles]
    pbest_fit = [evaluate_fitness(p, depot, capacity, dist)[0] for p in particles]
    gbest_idx = int(np.argmin(pbest_fit))
    gbest = pbest[gbest_idx][:]
    gbest_fit = pbest_fit[gbest_idx]

    history = []
    start_t = time.time()

    for it in range(n_iter):
        for i in range(n_particles):
            # swap-based move
            if random.random() < 0.7:
                particles[i] = swap_operator(particles[i])
            if random.random() < 0.3:
                particles[i] = swap_operator(particles[i])
            fit, _, _ = evaluate_fitness(particles[i], depot, capacity, dist)
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = particles[i][:]
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest = particles[i][:]
        history.append(gbest_fit)
        if progress_cb:
            progress_cb(it + 1, n_iter, gbest_fit)

    runtime = time.time() - start_t
    routes = decode_particle_to_routes(gbest, depot, capacity)
    total_fit, total_dist, n_vehicles = evaluate_fitness(gbest, depot, capacity, dist)
    return routes, total_dist, n_vehicles, history, runtime
