from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .data_preprocessing import Dataset, inverse_scale
from .lstm_model import LSTMConfig, NumpyLSTMRegressor
from .metrics import mae, mape, r2_score, rmse


BATCH_CHOICES = [16, 32, 64, 128]


def decode_solution(z: np.ndarray) -> dict[str, float | int]:
    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
    lr = 10 ** (math.log10(1e-4) + z[0] * (math.log10(1e-2) - math.log10(1e-4)))
    hidden_units = int(round(16 + z[1] * (128 - 16)))
    dropout = float(z[2] * 0.5)
    idx = int(round(z[3] * (len(BATCH_CHOICES) - 1)))
    idx = max(0, min(idx, len(BATCH_CHOICES) - 1))
    return {
        "learning_rate": float(lr),
        "hidden_units": hidden_units,
        "dropout": dropout,
        "batch_size": BATCH_CHOICES[idx],
    }


def cache_key(params: dict[str, float | int]) -> tuple[float, int, float, int]:
    return (
        round(float(params["learning_rate"]), 8),
        int(params["hidden_units"]),
        round(float(params["dropout"]), 4),
        int(params["batch_size"]),
    )


@dataclass
class LSTMObjective:
    dataset: Dataset
    epochs: int = 10
    seed: int = 1
    cache: dict[tuple[float, int, float, int], float] = field(default_factory=dict)
    records: list[dict[str, object]] = field(default_factory=list)

    def __call__(self, z: np.ndarray) -> float:
        params = decode_solution(z)
        key = cache_key(params)
        if key in self.cache:
            return self.cache[key]
        started = time.perf_counter()
        model = self._train_model(params)
        pred = model.predict(self.dataset.X_val, batch_size=512)
        score = rmse(self.dataset.y_val, pred)
        elapsed = time.perf_counter() - started
        self.cache[key] = score
        self.records.append(
            {
                "rmse_val": score,
                "time_sec": elapsed,
                "params_json": json.dumps(params, sort_keys=True),
                **params,
            }
        )
        return score

    def _train_model(self, params: dict[str, float | int]) -> NumpyLSTMRegressor:
        model = NumpyLSTMRegressor(
            LSTMConfig(
                hidden_units=int(params["hidden_units"]),
                dropout=float(params["dropout"]),
                learning_rate=float(params["learning_rate"]),
                seed=self.seed,
            )
        )
        model.fit(
            self.dataset.X_train,
            self.dataset.y_train,
            self.dataset.X_val,
            self.dataset.y_val,
            epochs=self.epochs,
            batch_size=int(params["batch_size"]),
        )
        return model

    def train_best_and_test(self, z: np.ndarray) -> dict[str, object]:
        params = decode_solution(z)
        X_trainval = np.vstack([self.dataset.X_train, self.dataset.X_val])
        y_trainval = np.vstack([self.dataset.y_train, self.dataset.y_val])
        model = NumpyLSTMRegressor(
            LSTMConfig(
                hidden_units=int(params["hidden_units"]),
                dropout=float(params["dropout"]),
                learning_rate=float(params["learning_rate"]),
                seed=self.seed + 10000,
            )
        )
        model.fit(
            X_trainval,
            y_trainval,
            self.dataset.X_val,
            self.dataset.y_val,
            epochs=self.epochs,
            batch_size=int(params["batch_size"]),
        )
        pred_scaled = model.predict(self.dataset.X_test, batch_size=512)
        y_true = inverse_scale(self.dataset.y_test, self.dataset.y_min, self.dataset.y_max)
        y_pred = inverse_scale(pred_scaled, self.dataset.y_min, self.dataset.y_max)
        return {
            "params": params,
            "rmse_test": rmse(y_true, y_pred),
            "mae_test": mae(y_true, y_pred),
            "mape_test": mape(y_true, y_pred),
            "r2_test": r2_score(y_true, y_pred),
            "y_true": y_true.reshape(-1),
            "y_pred": y_pred.reshape(-1),
        }
