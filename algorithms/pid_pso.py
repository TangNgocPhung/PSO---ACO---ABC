"""PID Controller Tuning với PSO."""
import numpy as np
import time
from scipy import signal

# Plant: G(s) = 1 / (s^3 + 6s^2 + 11s + 6)
NUM_PLANT = [1.0]
DEN_PLANT = [1.0, 6.0, 11.0, 6.0]
T_FINAL = 10.0
N_POINTS = 1000


def simulate_pid(Kp, Ki, Kd, setpoint=1.0):
    """Mô phỏng vòng đóng PID + plant + setpoint step."""
    # PID: C(s) = Kp + Ki/s + Kd*s ≈ (Kd s^2 + Kp s + Ki) / s
    pid_num = [Kd, Kp, Ki]
    pid_den = [1, 0]
    # Series: C(s)*P(s)
    series_num = np.polymul(pid_num, NUM_PLANT)
    series_den = np.polymul(pid_den, DEN_PLANT)
    # Closed-loop: G/(1+G)
    cl_num = series_num
    cl_den = np.polyadd(series_num, series_den)
    sys = signal.TransferFunction(cl_num, cl_den)
    t = np.linspace(0, T_FINAL, N_POINTS)
    t, y = signal.step(sys, T=t)
    return t, y * setpoint


def performance_metrics(t, y, setpoint=1.0):
    err = setpoint - y
    iae = np.trapezoid(np.abs(err), t)
    ise = np.trapezoid(err ** 2, t)
    itae = np.trapezoid(t * np.abs(err), t)
    # overshoot
    peak = np.max(y)
    overshoot = max(0, (peak - setpoint) / setpoint * 100)
    # rise time (10%→90%)
    try:
        i10 = np.where(y >= 0.1 * setpoint)[0][0]
        i90 = np.where(y >= 0.9 * setpoint)[0][0]
        rise = t[i90] - t[i10]
    except IndexError:
        rise = T_FINAL
    # settling time (±2%)
    band = 0.02 * setpoint
    settle = T_FINAL
    for i in range(len(t) - 1, -1, -1):
        if abs(y[i] - setpoint) > band:
            settle = t[i]
            break
    return {"IAE": iae, "ISE": ise, "ITAE": itae,
            "Overshoot%": overshoot, "Rise": rise, "Settle": settle}


def compute_fitness(Kp, Ki, Kd, metric="ITAE"):
    try:
        t, y = simulate_pid(Kp, Ki, Kd)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return 1e6
        m = performance_metrics(t, y)
        # phạt nếu overshoot quá lớn
        return m[metric] + 0.01 * m["Overshoot%"]
    except Exception:
        return 1e6


def pso_pid(n_particles=30, n_iter=50, w=0.7, c1=1.5, c2=1.5,
            bounds=None, metric="ITAE", seed=42, progress_cb=None):
    np.random.seed(seed)
    if bounds is None:
        bounds = np.array([[0.1, 50], [0, 20], [0, 20]])
    D = 3
    lb, ub = bounds[:, 0], bounds[:, 1]
    X = np.random.uniform(lb, ub, size=(n_particles, D))
    V = np.zeros_like(X)
    pbest = X.copy()
    pbest_fit = np.array([compute_fitness(*x, metric=metric) for x in X])
    gbest = pbest[np.argmin(pbest_fit)].copy()
    gbest_fit = pbest_fit.min()
    history = []
    start_t = time.time()

    for it in range(n_iter):
        for i in range(n_particles):
            r1, r2 = np.random.rand(D), np.random.rand(D)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            X[i] = np.clip(X[i] + V[i], lb, ub)
            f = compute_fitness(*X[i], metric=metric)
            if f < pbest_fit[i]:
                pbest_fit[i] = f
                pbest[i] = X[i].copy()
                if f < gbest_fit:
                    gbest_fit = f
                    gbest = X[i].copy()
        history.append(gbest_fit)
        if progress_cb:
            progress_cb(it + 1, n_iter, gbest_fit)

    runtime = time.time() - start_t
    return gbest, gbest_fit, history, runtime
