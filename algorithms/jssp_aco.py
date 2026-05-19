"""JSSP – Ant Colony Optimization (dựa trên benchmark FT06)."""
import numpy as np
import random
import time


# FT06 benchmark: 6 jobs × 6 machines, known optimum makespan = 55
FT06 = [
    [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
    [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
    [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
    [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
    [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
    [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
]
FT06_OPT = 55


def decode_schedule(sequence, jobs_data, num_machines):
    """Decode chuỗi job thành lịch (machine_id, job, op_idx, start, end)."""
    num_jobs = len(jobs_data)
    job_op_idx = [0] * num_jobs
    job_end_time = [0] * num_jobs
    mach_end_time = [0] * num_machines
    schedule = []
    for job in sequence:
        if job_op_idx[job] >= len(jobs_data[job]):
            continue
        m, dur = jobs_data[job][job_op_idx[job]]
        start = max(job_end_time[job], mach_end_time[m])
        end = start + dur
        schedule.append((m, job, job_op_idx[job], start, end))
        job_end_time[job] = end
        mach_end_time[m] = end
        job_op_idx[job] += 1
    makespan = max(mach_end_time) if mach_end_time else 0
    return schedule, makespan


def aco_jssp(jobs_data, num_ants=30, num_iter=100, alpha=1.0, beta=2.0,
             rho=0.2, Q=100.0, seed=42, progress_cb=None):
    random.seed(seed)
    np.random.seed(seed)
    num_jobs = len(jobs_data)
    num_ops = sum(len(j) for j in jobs_data)
    num_machines = max(m for job in jobs_data for m, _ in job) + 1

    # pheromone[op_position][job]
    pher = np.ones((num_ops, num_jobs))
    best_seq, best_sched, best_mk = None, None, float("inf")
    history = []
    start_t = time.time()

    for it in range(num_iter):
        for _ in range(num_ants):
            op_count = [0] * num_jobs
            sequence = []
            for pos in range(num_ops):
                cand = [j for j in range(num_jobs) if op_count[j] < len(jobs_data[j])]
                # heuristic: ưu tiên job có ít thao tác còn lại nhỏ hơn
                heur = np.array([1.0 / (len(jobs_data[j]) - op_count[j] + 0.01) for j in cand])
                tau = pher[pos, cand]
                probs = (tau ** alpha) * (heur ** beta)
                s = probs.sum()
                probs = probs / s if s > 0 else np.ones(len(cand)) / len(cand)
                chosen = np.random.choice(cand, p=probs)
                sequence.append(int(chosen))
                op_count[chosen] += 1

            sched, mk = decode_schedule(sequence, jobs_data, num_machines)
            if mk < best_mk:
                best_mk, best_seq, best_sched = mk, sequence[:], sched

        # Cập nhật pheromone
        pher *= (1 - rho)
        for pos, job in enumerate(best_seq):
            pher[pos, job] += Q / best_mk

        history.append(best_mk)
        if progress_cb:
            progress_cb(it + 1, num_iter, best_mk)

    runtime = time.time() - start_t
    return best_seq, best_sched, best_mk, history, runtime
