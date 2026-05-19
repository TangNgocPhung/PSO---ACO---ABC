"""Portfolio Optimization (Markowitz) – ACO."""
import numpy as np
import random
import time


STOCK_NAMES = ["VCB", "VNM", "HPG", "VIC", "MSN", "FPT", "MWG", "GAS", "VRE", "SAB"]
SECTORS = ["Ngân hàng", "Tiêu dùng", "Thép", "BĐS", "Tiêu dùng",
           "Công nghệ", "Bán lẻ", "Dầu khí", "BĐS", "Tiêu dùng"]
EXPECTED_RETURNS = np.array([0.12, 0.10, 0.15, 0.08, 0.11,
                              0.18, 0.14, 0.09, 0.10, 0.13])
STD_DEVS = np.array([0.18, 0.15, 0.28, 0.22, 0.20,
                     0.32, 0.25, 0.19, 0.21, 0.17])
CORRELATION = np.array([
    [1.00, 0.30, 0.25, 0.45, 0.30, 0.20, 0.35, 0.50, 0.40, 0.30],
    [0.30, 1.00, 0.20, 0.25, 0.60, 0.15, 0.40, 0.20, 0.20, 0.55],
    [0.25, 0.20, 1.00, 0.30, 0.25, 0.20, 0.30, 0.35, 0.30, 0.20],
    [0.45, 0.25, 0.30, 1.00, 0.25, 0.30, 0.30, 0.30, 0.65, 0.25],
    [0.30, 0.60, 0.25, 0.25, 1.00, 0.20, 0.50, 0.20, 0.25, 0.55],
    [0.20, 0.15, 0.20, 0.30, 0.20, 1.00, 0.30, 0.20, 0.25, 0.15],
    [0.35, 0.40, 0.30, 0.30, 0.50, 0.30, 1.00, 0.25, 0.30, 0.40],
    [0.50, 0.20, 0.35, 0.30, 0.20, 0.20, 0.25, 1.00, 0.30, 0.20],
    [0.40, 0.20, 0.30, 0.65, 0.25, 0.25, 0.30, 0.30, 1.00, 0.25],
    [0.30, 0.55, 0.20, 0.25, 0.55, 0.15, 0.40, 0.20, 0.25, 1.00],
])
COV_MATRIX = np.outer(STD_DEVS, STD_DEVS) * CORRELATION
N = len(STOCK_NAMES)
RF = 0.05  # risk-free


def portfolio_return(w, ret=EXPECTED_RETURNS):
    return float(np.dot(w, ret))


def portfolio_risk(w, cov=COV_MATRIX):
    return float(np.sqrt(np.dot(w, np.dot(cov, w))))


def sharpe_ratio(w, ret=EXPECTED_RETURNS, cov=COV_MATRIX, rf=RF):
    r = portfolio_return(w, ret)
    s = portfolio_risk(w, cov)
    return (r - rf) / s if s > 0 else 0


def normalize_weights(w):
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def aco_portfolio(n_ants=30, n_iter=200, alpha=1.0, beta=2.0, rho=0.10,
                  Q=1.0, num_levels=21, top_frac=0.3, seed=42, progress_cb=None):
    random.seed(seed)
    np.random.seed(seed)
    levels = np.linspace(0.0, 0.5, num_levels)
    K = num_levels
    pher = np.ones((N, K))
    history = []
    best_w, best_sharpe = None, -float("inf")
    start_t = time.time()

    for it in range(n_iter):
        ants = []
        for _ in range(n_ants):
            chosen_levels = np.zeros(N)
            for i in range(N):
                tau = pher[i]
                heur = np.ones(K)
                p = (tau ** alpha) * (heur ** beta)
                s = p.sum()
                p = p / s if s > 0 else np.ones(K) / K
                chosen_levels[i] = levels[np.random.choice(K, p=p)]
            w = normalize_weights(chosen_levels + 0.01)
            sr = sharpe_ratio(w)
            ants.append((w, sr))
            if sr > best_sharpe:
                best_sharpe, best_w = sr, w.copy()

        # top fraction reinforcement
        ants.sort(key=lambda a: -a[1])
        top_n = max(1, int(len(ants) * top_frac))
        pher *= (1 - rho)
        for w, sr in ants[:top_n]:
            for i in range(N):
                # tìm level gần với weight nhất
                level_idx = int(np.argmin(np.abs(levels - w[i])))
                pher[i, level_idx] += Q * max(sr, 0.01)

        history.append(best_sharpe)
        if progress_cb:
            progress_cb(it + 1, n_iter, best_sharpe)

    runtime = time.time() - start_t
    return best_w, best_sharpe, history, runtime
