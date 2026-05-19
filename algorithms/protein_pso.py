"""Protein Folding – HP Model 2D với PSO."""
import numpy as np
import random
import time


DATASETS = {
    "HP-20": ("HPHPPHHPHPPHPHHPPHPH", -9),
    "HP-24": ("HHPPHPPHPPHPPHPPHPPHPPHH", -9),
    "HP-36": ("PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP", -14),
    "HP-48": ("PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH", -23),
}

# Direction: 0=right, 1=up, 2=left, 3=down
DIRS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def decode(directions):
    coords = [(0, 0)]
    for d in directions:
        x, y = coords[-1]
        dx, dy = DIRS[int(d)]
        coords.append((x + dx, y + dy))
    return coords


def fitness(seq, directions, h_indices, penalty_weight=40):
    coords = decode(directions)
    seen = {}
    overlap = 0
    for i, c in enumerate(coords):
        if c in seen:
            overlap += 1
        else:
            seen[c] = i
    if overlap > 0:
        return penalty_weight * overlap, 0, penalty_weight * overlap

    energy = 0
    pos_map = {c: i for i, c in enumerate(coords)}
    for i in h_indices:
        x, y = coords[i]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (x + dx, y + dy)
            if nb in pos_map:
                j = pos_map[nb]
                if j in h_indices and abs(i - j) > 1:
                    energy -= 0.5  # cộng đôi
    return -energy, energy, 0  # fitness = -E (minimize), energy=true H-H contacts


def pso_protein(seq, n_particles=40, n_iter=150, w_start=0.9, w_end=0.4,
                c1=1.5, c2=1.5, penalty_weight=40, seed=42, progress_cb=None):
    random.seed(seed)
    np.random.seed(seed)
    h_indices = {i for i, aa in enumerate(seq) if aa == "H"}
    L = len(seq) - 1  # số hướng

    particles = np.random.randint(0, 4, size=(n_particles, L))
    velocities = np.random.uniform(-1, 1, size=(n_particles, L))
    pbest = particles.copy()
    pbest_fit = np.array([fitness(seq, p, h_indices, penalty_weight)[0] for p in particles])
    gbest_idx = int(np.argmin(pbest_fit))
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    history_fit, history_energy = [], []
    start_t = time.time()

    for it in range(n_iter):
        w = w_start - (w_start - w_end) * it / max(1, n_iter - 1)
        for i in range(n_particles):
            r1, r2 = np.random.rand(L), np.random.rand(L)
            velocities[i] = (w * velocities[i]
                             + c1 * r1 * (pbest[i] - particles[i])
                             + c2 * r2 * (gbest - particles[i]))
            # cập nhật + clip vào {0,1,2,3}
            new_pos = particles[i] + velocities[i]
            particles[i] = np.clip(np.round(new_pos).astype(int), 0, 3)
            fit, energy, pen = fitness(seq, particles[i], h_indices, penalty_weight)
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = particles[i].copy()
                if fit < gbest_fit:
                    gbest_fit, gbest = fit, particles[i].copy()
        _, eng, _ = fitness(seq, gbest, h_indices, penalty_weight)
        history_fit.append(gbest_fit)
        history_energy.append(eng)
        if progress_cb:
            progress_cb(it + 1, n_iter, gbest_fit)

    runtime = time.time() - start_t
    return gbest, gbest_fit, history_fit, history_energy, runtime
