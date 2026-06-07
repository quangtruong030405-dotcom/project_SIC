from __future__ import annotations

import numpy as np


def _weighted_field(current: np.ndarray, points: np.ndarray, weights: np.ndarray, sign: float, eps: float) -> np.ndarray:
    if len(points) == 0:
        return np.zeros_like(current)
    diff = sign * (points - current)
    denom = np.linalg.norm(diff, axis=1, keepdims=True) + eps
    weighted = weights[:, None] * diff / denom
    return np.sum(weighted, axis=0) / max(float(np.sum(weights)), eps)


def _select_void(rng: np.random.Generator, archive_z: np.ndarray, archive_f: np.ndarray, k: int, m: int = 5) -> np.ndarray:
    candidates = rng.random((k, archive_z.shape[1]))
    scores = []
    best_f = float(np.min(archive_f))
    for y in candidates:
        dist = np.linalg.norm(archive_z - y, axis=1)
        order = np.argsort(dist)[: min(m, len(dist))]
        d = float(np.min(dist))
        f_hat = float(np.mean(archive_f[order]))
        scores.append((d + 1e-9) / (max(f_hat - best_f, 0.0) + 1e-4))
    return candidates[int(np.argmax(scores))]


def optimize_mavo(
    objective,
    population: int,
    budget: int,
    dim: int = 4,
    seed: int = 1,
    elite_ratio: float = 0.2,
    bad_ratio: float = 0.2,
    void_candidates: int = 64,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    z = rng.random((population, dim))
    eta = rng.uniform(0.04, 0.12, size=population)
    fitness = np.full(population, np.inf)
    archive_z, archive_f, rows = [], [], []
    best_z = z[0].copy()
    best_f = float("inf")
    nfe = 0

    for i in range(population):
        f = float(objective(z[i]))
        fitness[i] = f
        archive_z.append(z[i].copy())
        archive_f.append(f)
        nfe += 1
        if f < best_f:
            best_f = f
            best_z = z[i].copy()
        rows.append({"nfe": nfe, "rmse_val": f, "best_so_far": best_f})
        if nfe >= budget:
            break

    t = 0
    total_steps = max(1, budget // max(population, 1))
    stale = 0
    while nfe < budget:
        az = np.asarray(archive_z)
        af = np.asarray(archive_f)
        order = np.argsort(af)
        elite_n = max(1, int(np.ceil(len(order) * elite_ratio)))
        bad_n = max(1, int(np.ceil(len(order) * bad_ratio)))
        elites = az[order[:elite_n]]
        bads = az[order[-bad_n:]]
        elite_weights = 1.0 / (np.arange(elite_n) + 1.0)
        bad_weights = (np.arange(bad_n) + 1.0) / bad_n
        y_void = _select_void(rng, az, af, void_candidates)

        tau = min(t / total_steps, 1.0)
        alpha = 0.35 + 0.45 * tau
        beta = 0.30
        gamma = 0.65 * (1.0 - tau) + 0.10
        sigma = 0.08 * (1.0 - tau)
        prev_best = best_f

        for i in range(population):
            memory = _weighted_field(z[i], elites, elite_weights, sign=1.0, eps=1e-9)
            antagonist = _weighted_field(z[i], bads, bad_weights, sign=-1.0, eps=1e-9)
            void = y_void - z[i]
            void /= np.linalg.norm(void) + 1e-9
            step = alpha * memory + beta * antagonist + gamma * void + sigma * rng.normal(size=dim)
            z_new = np.clip(z[i] + eta[i] * step, 0.0, 1.0)
            f_new = float(objective(z_new))
            archive_z.append(z_new.copy())
            archive_f.append(f_new)
            nfe += 1
            if f_new < fitness[i]:
                z[i] = z_new
                fitness[i] = f_new
                eta[i] = min(0.30, 1.15 * eta[i])
            else:
                eta[i] = max(0.025, 0.65 * eta[i])
            if f_new < best_f:
                best_f = f_new
                best_z = z_new.copy()
            rows.append({"nfe": nfe, "rmse_val": f_new, "best_so_far": best_f})
            if nfe >= budget:
                break

        diversity = float(np.mean(np.linalg.norm(z - np.mean(z, axis=0), axis=1)))
        stale = stale + 1 if best_f >= prev_best - 1e-8 else 0
        if diversity < 0.05 and stale >= 3 and nfe < budget:
            worst = np.argsort(fitness)[-max(1, population // 5) :]
            for idx in worst:
                z[idx] = np.clip(y_void + 0.05 * rng.normal(size=dim), 0.0, 1.0)
                fitness[idx] = float("inf")
            stale = 0
        t += 1

    return {"best_z": best_z, "best_f": best_f, "history": rows}
