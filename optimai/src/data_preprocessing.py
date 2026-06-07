from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_min: float
    y_max: float
    hourly: pd.Series


def _make_sequences(values: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(look_back, len(values)):
        X.append(values[i - look_back : i])
        y.append(values[i])
    X_arr = np.asarray(X, dtype=np.float32)[:, :, None]
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X_arr, y_arr


def load_power_dataset(
    csv_path: str,
    look_back: int = 24,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_hours: int | None = None,
) -> Dataset:
    df = pd.read_csv(csv_path, na_values=["?"])
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    power = pd.to_numeric(df["Global_active_power"], errors="coerce")
    hourly = (
        pd.Series(power.to_numpy(dtype=float), index=dt)
        .sort_index()
        .resample("h")
        .mean()
        .interpolate(method="time")
        .ffill()
        .bfill()
    )
    if max_hours is not None:
        hourly = hourly.iloc[:max_hours]

    n = len(hourly)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_values = hourly.iloc[:train_end].to_numpy(dtype=np.float32)
    y_min = float(np.min(train_values))
    y_max = float(np.max(train_values))
    scale = max(y_max - y_min, 1e-8)
    values = ((hourly.to_numpy(dtype=np.float32) - y_min) / scale).astype(np.float32)

    X_all, y_all = _make_sequences(values, look_back)
    train_seq_end = max(train_end - look_back, 1)
    val_seq_end = max(val_end - look_back, train_seq_end + 1)

    return Dataset(
        X_train=X_all[:train_seq_end],
        y_train=y_all[:train_seq_end],
        X_val=X_all[train_seq_end:val_seq_end],
        y_val=y_all[train_seq_end:val_seq_end],
        X_test=X_all[val_seq_end:],
        y_test=y_all[val_seq_end:],
        y_min=y_min,
        y_max=y_max,
        hourly=hourly,
    )


def inverse_scale(values: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    return np.asarray(values, dtype=float) * (y_max - y_min) + y_min
