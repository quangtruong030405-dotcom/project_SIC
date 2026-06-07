from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_preprocessing import load_power_dataset
from src.objective import LSTMObjective, decode_solution
from src.optimizers.mavo import optimize_mavo
from src.optimizers.pso import optimize_pso
from src.optimizers.random_search import random_search
from src.utils import ensure_dir, simple_svg_actual_predicted, simple_svg_histogram, simple_svg_line_chart, write_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAVO-LSTM vs PSO-LSTM experiment")
    parser.add_argument("--csv", default="household_power_consumption.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--look-back", type=int, default=24)
    parser.add_argument("--max-hours", type=int, default=2400)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def summarize_validation(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for algo in sorted({str(r["algorithm"]) for r in rows}):
        vals = [float(r["best_rmse_val"]) for r in rows if r["algorithm"] == algo]
        out.append(
            {
                "algorithm": algo,
                "best_rmse_val": min(vals),
                "mean_rmse_val": statistics.fmean(vals),
                "std_rmse_val": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "worst_rmse_val": max(vals),
            }
        )
    return out


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> dict[str, float]:
    diffs = [a - b for a, b in zip(x, y) if abs(a - b) > 1e-12]
    n = len(diffs)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "w_minus": 0.0, "p_value_approx": 1.0}
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    rank = 1
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(diffs[order[j + 1]]) - abs(diffs[order[i]])) < 1e-12:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        rank += j - i + 1
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, diffs) if d < 0)
    w = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w - mean) / math.sqrt(var) if var > 0 else 0.0
    p = 2.0 * min(_normal_cdf(z), 1.0 - _normal_cdf(z))
    return {"n": n, "w_plus": w_plus, "w_minus": w_minus, "p_value_approx": max(0.0, min(1.0, p))}


def statistical_tests(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_algo = {}
    for row in rows:
        by_algo.setdefault(str(row["algorithm"]), {})[int(row["seed"])] = float(row["best_rmse_val"])
    tests = []
    pairs = [("MAVO-LSTM", "PSO-LSTM"), ("MAVO-LSTM", "Random Search"), ("PSO-LSTM", "Random Search")]
    for left, right in pairs:
        common = sorted(set(by_algo.get(left, {})) & set(by_algo.get(right, {})))
        if not common:
            continue
        result = wilcoxon_signed_rank([by_algo[left][s] for s in common], [by_algo[right][s] for s in common])
        tests.append({"comparison": f"{left} vs {right}", **result})
    return tests


def main() -> None:
    args = parse_args()
    if args.quick:
        args.seeds = [1]
        args.population = 4
        args.budget = 12
        args.epochs = 2
        args.max_hours = 700

    results_dir = ensure_dir(args.results_dir)
    fig_dir = ensure_dir(results_dir / "figures")
    dataset = load_power_dataset(args.csv, look_back=args.look_back, max_hours=args.max_hours)

    optimizers = {
        "MAVO-LSTM": lambda obj, seed: optimize_mavo(obj, args.population, args.budget, seed=seed),
        "PSO-LSTM": lambda obj, seed: optimize_pso(obj, args.population, args.budget, seed=seed),
        "Random Search": lambda obj, seed: random_search(obj, args.budget, seed=seed),
    }

    best_rows = []
    test_rows = []
    first_seed_curves = {}
    for seed in args.seeds:
        for name, runner in optimizers.items():
            objective = LSTMObjective(dataset=dataset, epochs=args.epochs, seed=seed)
            result = runner(objective, seed)
            history = result["history"]
            safe_name = name.lower().replace(" ", "_").replace("-", "_")
            write_rows(results_dir / f"{safe_name}_convergence_seed_{seed:02d}.csv", history)
            if seed == args.seeds[0]:
                first_seed_curves[name] = [float(row["best_so_far"]) for row in history]

            params = decode_solution(result["best_z"])
            best_rows.append(
                {
                    "algorithm": name,
                    "seed": seed,
                    "best_rmse_val": float(result["best_f"]),
                    **params,
                }
            )
            test = objective.train_best_and_test(result["best_z"])
            test_rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "rmse_test": test["rmse_test"],
                    "mae_test": test["mae_test"],
                    "mape_test": test["mape_test"],
                    "r2_test": test["r2_test"],
                    **test["params"],
                }
            )
            if seed == args.seeds[0]:
                pred_rows = [
                    {"index": i, "actual": float(a), "predicted": float(p)}
                    for i, (a, p) in enumerate(zip(test["y_true"], test["y_pred"]))
                ]
                write_rows(results_dir / f"prediction_{safe_name}_test.csv", pred_rows)
                if name == "MAVO-LSTM":
                    simple_svg_actual_predicted(fig_dir / "actual_vs_predicted.svg", list(test["y_true"]), list(test["y_pred"]))
                    errors = list(np.asarray(test["y_true"]) - np.asarray(test["y_pred"]))
                    simple_svg_histogram(fig_dir / "error_distribution.svg", errors)

        default_objective = LSTMObjective(dataset=dataset, epochs=args.epochs, seed=seed)
        default_z = np.array([0.5, (32 - 16) / (128 - 16), 0.2 / 0.5, 1 / 3], dtype=float)
        default_val = default_objective(default_z)
        default_test = default_objective.train_best_and_test(default_z)
        best_rows.append(
            {
                "algorithm": "Default LSTM",
                "seed": seed,
                "best_rmse_val": default_val,
                **decode_solution(default_z),
            }
        )
        test_rows.append(
            {
                "model": "Default LSTM",
                "seed": seed,
                "rmse_test": default_test["rmse_test"],
                "mae_test": default_test["mae_test"],
                "mape_test": default_test["mape_test"],
                "r2_test": default_test["r2_test"],
                **default_test["params"],
            }
        )

    write_rows(results_dir / "best_params.csv", best_rows)
    write_rows(results_dir / "validation_summary.csv", summarize_validation(best_rows))
    write_rows(results_dir / "test_summary.csv", test_rows)
    write_rows(results_dir / "statistical_tests.csv", statistical_tests(best_rows))
    simple_svg_line_chart(fig_dir / "convergence_curve.svg", first_seed_curves)

    print("Experiment complete")
    print(f"Hourly points: {len(dataset.hourly)}")
    print(pd.DataFrame(summarize_validation(best_rows)).to_string(index=False))
    print(f"Results: {Path(results_dir).resolve()}")


if __name__ == "__main__":
    main()
