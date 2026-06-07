from __future__ import annotations

import numpy as np


def optimize_pso(
    objective,
    population: int,
    budget: int,
    dim: int = 4,
    seed: int = 1,
    vmax: float = 0.25,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    z = rng.random((population, dim))
    v = rng.uniform(-0.05, 0.05, size=(population, dim))
    pbest = z.copy()
    pbest_f = np.full(population, np.inf)
    rows = []
    nfe = 0
    for i in range(population):
        pbest_f[i] = float(objective(z[i]))
        nfe += 1
        rows.append({"nfe": nfe, "rmse_val": pbest_f[i], "best_so_far": float(np.min(pbest_f))})
        if nfe >= budget:
            break
    gbest = pbest[int(np.argmin(pbest_f))].copy()
    gbest_f = float(np.min(pbest_f))
    t = 0
    total_steps = max(1, budget // max(population, 1))
    while nfe < budget:
        omega = 0.9 - (0.5 * min(t, total_steps) / total_steps)
        for i in range(population):
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            v[i] = omega * v[i] + 1.5 * r1 * (pbest[i] - z[i]) + 1.5 * r2 * (gbest - z[i])
            v[i] = np.clip(v[i], -vmax, vmax)
            z[i] = np.clip(z[i] + v[i], 0.0, 1.0)
            f = float(objective(z[i]))
            nfe += 1
            if f < pbest_f[i]:
                pbest_f[i] = f
                pbest[i] = z[i].copy()
            if f < gbest_f:
                gbest_f = f
                gbest = z[i].copy()
            rows.append({"nfe": nfe, "rmse_val": f, "best_so_far": gbest_f})
            if nfe >= budget:
                break
        t += 1
    return {"best_z": gbest, "best_f": gbest_f, "history": rows}
