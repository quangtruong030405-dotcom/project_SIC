from __future__ import annotations

import numpy as np


def random_search(objective, budget: int, dim: int = 4, seed: int = 1) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    best_z = None
    best_f = float("inf")
    rows = []
    for nfe in range(1, budget + 1):
        z = rng.random(dim)
        f = float(objective(z))
        if f < best_f:
            best_f = f
            best_z = z.copy()
        rows.append({"nfe": nfe, "rmse_val": f, "best_so_far": best_f})
    return {"best_z": best_z, "best_f": best_f, "history": rows}
