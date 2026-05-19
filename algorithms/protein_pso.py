"""
Protein Folding – HP Model 2D & 3D với PSO.

Quy ước:
    - energy = số cặp H-H tiếp xúc (KHÔNG kề trên chuỗi nhưng kề trên lưới)
              => giá trị nguyên dương; càng LỚN càng tốt.
    - fitness = -energy + penalty_overlap   => MINIMIZE.
    - mode = "2D" (4 hướng)  hoặc  "3D" (6 hướng).
"""
import numpy as np
import random
import time

# ---------------------------------------------------------------------------
#  HƯỚNG DI CHUYỂN
# ---------------------------------------------------------------------------
DIRS_2D = {
    0: (1, 0),    # +x
    1: (0, 1),    # +y
    2: (-1, 0),   # -x
    3: (0, -1),   # -y
}

DIRS_3D = {
    0: (1, 0, 0),    # +x
    1: (-1, 0, 0),   # -x
    2: (0, 1, 0),    # +y
    3: (0, -1, 0),   # -y
    4: (0, 0, 1),    # +z
    5: (0, 0, -1),   # -z
}

NEIGHBORS_2D = [(1, 0), (-1, 0), (0, 1), (0, -1)]
NEIGHBORS_3D = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                (0, -1, 0), (0, 0, 1), (0, 0, -1)]

# ---------------------------------------------------------------------------
#  DATASET HP (số cặp H-H tối ưu đã biết – giá trị dương)
# ---------------------------------------------------------------------------
DATASETS_2D = {
    "HP-20": ("HPHPPHHPHPPHPHHPPHPH", 9),
    "HP-24": ("HHPPHPPHPPHPPHPPHPPHPPHH", 9),
    "HP-36": ("PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP", 14),
    "HP-48": ("PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH", 23),
}

DATASETS_3D = {
    "HP-20": ("HPHPPHHPHPPHPHHPPHPH", 11),
    "HP-24": ("HHPPHPPHPPHPPHPPHPPHPPHH", 13),
    "HP-36": ("PPHPPHHPPPPHHPPPPHHPPPPHHPPPPHHPPPPHH", 17),
    "HP-48": ("PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP", 25),
}


# ---------------------------------------------------------------------------
#  DECODE: chuỗi hướng -> toạ độ lưới
# ---------------------------------------------------------------------------
def decode(directions, mode="2D"):
    """Trả về danh sách tuple toạ độ (2D hoặc 3D)."""
    dir_map = DIRS_2D if mode == "2D" else DIRS_3D
    origin = (0, 0) if mode == "2D" else (0, 0, 0)
    coords = [origin]
    for d in directions:
        last = coords[-1]
        delta = dir_map[int(d) % len(dir_map)]
        coords.append(tuple(a + b for a, b in zip(last, delta)))
    return coords


# ---------------------------------------------------------------------------
#  FITNESS: -contacts + penalty_overlap
# ---------------------------------------------------------------------------
def get_fitness_details(seq, directions, h_indices, mode="2D", penalty_weight=40):
    """Trả về (fitness, energy, penalty). fitness càng nhỏ càng tốt."""
    coords = decode(directions, mode)
    n = len(coords)
    overlap = n - len(set(coords))
    if overlap > 0:
        return float(penalty_weight * overlap), 0, penalty_weight * overlap

    neighbors = NEIGHBORS_2D if mode == "2D" else NEIGHBORS_3D
    pos_map = {c: i for i, c in enumerate(coords)}
    contacts = 0
    for i in h_indices:
        c = coords[i]
        for d in neighbors:
            nb = tuple(a + b for a, b in zip(c, d))
            if nb in pos_map:
                j = pos_map[nb]
                if j in h_indices and j > i + 1:
                    contacts += 1
    # contacts = số cặp (mỗi cặp đếm 1 lần do j > i+1)
    return float(-contacts), contacts, 0


def fitness(seq, directions, h_indices, mode="2D", penalty_weight=40):
    """Wrapper trả về chỉ giá trị fitness (để PSO so sánh)."""
    f, _, _ = get_fitness_details(seq, directions, h_indices, mode, penalty_weight)
    return f


# ---------------------------------------------------------------------------
#  PSO MAIN LOOP
# ---------------------------------------------------------------------------
def pso_protein(seq, mode="2D", n_particles=80, n_iter=200,
                w_start=0.9, w_end=0.4, c1=1.5, c2=1.5,
                penalty_weight=40, early_stop=40, seed=42, progress_cb=None):
    """
    Trả về:
        gbest          : np.ndarray các hướng tối ưu
        gbest_fit      : float (càng nhỏ càng tốt)
        history_fit    : list float
        history_energy : list int (số cặp H-H, càng lớn càng tốt)
        runtime        : float (giây)
    """
    random.seed(seed)
    np.random.seed(seed)

    h_indices = {i for i, aa in enumerate(seq) if aa == "H"}
    num_dirs = 4 if mode == "2D" else 6
    L = len(seq) - 1   # số hướng

    particles = np.random.randint(0, num_dirs, size=(n_particles, L))
    velocities = np.zeros((n_particles, L))

    pbest = particles.copy()
    pbest_fit = np.array([
        fitness(seq, p, h_indices, mode, penalty_weight) for p in particles
    ])
    g_idx = int(np.argmin(pbest_fit))
    gbest = pbest[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])

    history_fit, history_energy = [], []
    no_improve = 0
    start = time.time()

    for it in range(n_iter):
        w = w_start - (w_start - w_end) * it / max(1, n_iter - 1)
        for i in range(n_particles):
            r1, r2 = np.random.rand(L), np.random.rand(L)
            velocities[i] = (w * velocities[i]
                             + c1 * r1 * (pbest[i] - particles[i])
                             + c2 * r2 * (gbest - particles[i]))
            new_pos = particles[i] + velocities[i]
            particles[i] = np.clip(np.round(new_pos).astype(int), 0, num_dirs - 1)
            f = fitness(seq, particles[i], h_indices, mode, penalty_weight)
            if f < pbest_fit[i]:
                pbest_fit[i] = f
                pbest[i] = particles[i].copy()
                if f < gbest_fit:
                    gbest_fit, gbest = f, particles[i].copy()
                    no_improve = 0

        no_improve += 1
        _, eng, _ = get_fitness_details(seq, gbest, h_indices, mode, penalty_weight)
        history_fit.append(gbest_fit)
        history_energy.append(eng)
        if progress_cb:
            progress_cb(it + 1, n_iter, gbest_fit)
        if no_improve > early_stop:
            break

    runtime = time.time() - start
    return gbest, gbest_fit, history_fit, history_energy, runtime
