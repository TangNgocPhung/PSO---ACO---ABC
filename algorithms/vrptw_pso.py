"""VRPTW – Vehicle Routing Problem with Time Windows (PSO chuẩn).

Khác biệt so với phiên bản trước:
    - Bổ sung **gbest-guided gene copy** (thành phần social của PSO thực thụ)
    - Bổ sung **pbest update** rõ ràng
    - Penalty coefficient cấu hình được (mặc định 1000 theo notebook gốc)
    - Hỗ trợ load **Solomon CSV** thật (auto-detect cột)
"""
import numpy as np
import random
import math
import time
import pandas as pd


# ---------------------------------------------------------------------------
#  TẠO DATASET
# ---------------------------------------------------------------------------
def make_solomon_like(n=20, capacity=200, seed=42):
    """Tạo dữ liệu kiểu Solomon (depot + khách)."""
    rng = np.random.default_rng(seed)
    customers = []
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


def load_solomon_csv(df, capacity=200, normalize=False):
    """Load Solomon CSV format (auto-detect cột).

    Format Solomon mặc định: CUST_NO, XCOORD, YCOORD, DEMAND, READY_TIME, DUE_DATE, SERVICE_TIME
    Một số biến thể dùng tên ngắn: X, Y, ID, READY, DUE, SERVICE.
    """
    cols = {c.lower().strip(): c for c in df.columns}

    def find(*keys):
        for k in keys:
            for col_lower, original in cols.items():
                if k in col_lower:
                    return original
        return None

    id_col = find("cust", "id")
    x_col = find("xcoord", "x")
    y_col = find("ycoord", "y")
    d_col = find("demand")
    r_col = find("ready")
    due_col = find("due")
    s_col = find("service")

    missing = [name for name, col in zip(
        ["X", "Y", "DEMAND", "READY_TIME", "DUE_DATE", "SERVICE_TIME"],
        [x_col, y_col, d_col, r_col, due_col, s_col]) if col is None]
    if missing:
        raise ValueError(f"Thiếu các cột: {missing}. Có sẵn: {list(df.columns)}")

    xs = df[x_col].values.astype(float)
    ys = df[y_col].values.astype(float)

    if normalize:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 100))
        coords = scaler.fit_transform(np.column_stack([xs, ys]))
        xs, ys = coords[:, 0], coords[:, 1]

    customers = []
    for i in range(len(df)):
        customers.append({
            "id": int(df[id_col].iloc[i]) if id_col else i,
            "x": float(xs[i]),
            "y": float(ys[i]),
            "demand": int(df[d_col].iloc[i]),
            "ready_time": float(df[r_col].iloc[i]),
            "due_date": float(df[due_col].iloc[i]),
            "service_time": float(df[s_col].iloc[i]),
        })
    # đảm bảo depot ở index 0
    if customers[0]["demand"] != 0:
        # tìm khách có demand=0 (depot) và đưa lên đầu
        depot_idx = next((i for i, c in enumerate(customers) if c["demand"] == 0), 0)
        customers = [customers[depot_idx]] + [c for i, c in enumerate(customers) if i != depot_idx]
    customers[0]["id"] = 0
    # đánh lại id 1..n cho các khách
    for k, c in enumerate(customers[1:], start=1):
        c["id"] = k
    return customers, capacity


# ---------------------------------------------------------------------------
#  HÀM PHỤ TRỢ
# ---------------------------------------------------------------------------
def build_distance_matrix(customers):
    n = len(customers)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = math.hypot(customers[i]["x"] - customers[j]["x"],
                                 customers[i]["y"] - customers[j]["y"])
    return d


def decode_particle_to_routes(particle, depot, capacity):
    """Chia chuỗi khách thành các route theo capacity."""
    routes = []
    cur, load = [], 0
    for c in particle:
        if load + c["demand"] > capacity:
            if cur:
                routes.append(cur)
            cur, load = [], 0
        cur.append(c)
        load += c["demand"]
    if cur:
        routes.append(cur)
    return routes


def evaluate_fitness(particle, depot, capacity, dist, penalty_coef=1000):
    """Trả về (fitness, total_dist, n_vehicles, n_late)."""
    routes = decode_particle_to_routes(particle, depot, capacity)
    total_dist, time_penalty, n_late = 0.0, 0.0, 0
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
                time_penalty += (time_now - c["due_date"])
                n_late += 1
            time_now += c["service_time"]
            prev_id = c["id"]
        total_dist += dist[prev_id, depot["id"]]
    vehicle_penalty = len(routes) * 50
    fitness = total_dist + penalty_coef * time_penalty + vehicle_penalty
    return fitness, total_dist, len(routes), n_late


def swap_operator(particle):
    new = particle[:]
    i, j = random.sample(range(len(new)), 2)
    new[i], new[j] = new[j], new[i]
    return new


def gbest_guided_move(particle, gbest, prob=0.5):
    """Sao chép 1 gene từ gbest sang particle (thành phần social của PSO).

    Đây chính là phần thiếu trước đây – biến swap-based search thành PSO thực thụ.
    """
    if random.random() > prob or len(particle) < 2:
        return particle
    # chọn vị trí ngẫu nhiên và "mượn" khách từ gbest tại vị trí đó
    pos = random.randrange(len(particle))
    target_cust = gbest[pos]
    # tìm vị trí của target_cust trong particle để swap về pos
    new = particle[:]
    cur_pos = next((i for i, c in enumerate(new) if c["id"] == target_cust["id"]), None)
    if cur_pos is not None and cur_pos != pos:
        new[pos], new[cur_pos] = new[cur_pos], new[pos]
    return new


# ---------------------------------------------------------------------------
#  PSO MAIN LOOP
# ---------------------------------------------------------------------------
def pso_vrptw(customers, capacity, n_particles=30, n_iter=80,
              penalty_coef=1000, gbest_prob=0.5, swap_prob=0.7,
              seed=42, progress_cb=None):
    """
    Trả về: routes, total_dist, n_vehicles, history, runtime, n_late
    """
    random.seed(seed)
    np.random.seed(seed)
    depot = customers[0]
    others = customers[1:]
    dist = build_distance_matrix(customers)

    # Khởi tạo
    particles = []
    for _ in range(n_particles):
        p = others[:]
        random.shuffle(p)
        particles.append(p)

    pbest = [p[:] for p in particles]
    pbest_fit = [evaluate_fitness(p, depot, capacity, dist, penalty_coef)[0] for p in particles]
    g_idx = int(np.argmin(pbest_fit))
    gbest = pbest[g_idx][:]
    gbest_fit = pbest_fit[g_idx]

    history = []
    start_t = time.time()

    for it in range(n_iter):
        for i in range(n_particles):
            # Bước 1: Swap-based local search (cognitive / personal exploration)
            if random.random() < swap_prob:
                particles[i] = swap_operator(particles[i])

            # Bước 2: Gbest-guided gene copy (social learning của PSO)
            particles[i] = gbest_guided_move(particles[i], gbest, gbest_prob)

            # Đánh giá fitness
            fit, _, _, _ = evaluate_fitness(particles[i], depot, capacity, dist, penalty_coef)

            # Cập nhật pbest
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = particles[i][:]

        # Cập nhật gbest
        g_idx = int(np.argmin(pbest_fit))
        if pbest_fit[g_idx] < gbest_fit:
            gbest_fit = pbest_fit[g_idx]
            gbest = pbest[g_idx][:]

        history.append(gbest_fit)
        if progress_cb:
            progress_cb(it + 1, n_iter, gbest_fit)

    runtime = time.time() - start_t
    routes = decode_particle_to_routes(gbest, depot, capacity)
    _, total_dist, n_vehicles, n_late = evaluate_fitness(gbest, depot, capacity, dist, penalty_coef)
    return routes, total_dist, n_vehicles, history, runtime, n_late
