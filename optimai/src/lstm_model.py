from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


@dataclass
class LSTMConfig:
    input_size: int = 1
    hidden_units: int = 32
    dropout: float = 0.1
    learning_rate: float = 1e-3
    seed: int = 1


class NumpyLSTMRegressor:
    """Small one-layer LSTM regressor with Adam, built for reproducible experiments."""

    def __init__(self, config: LSTMConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        h = config.hidden_units
        fan = config.input_size + h
        limit = math.sqrt(1.0 / fan)
        self.W = self.rng.uniform(-limit, limit, size=(config.input_size + h, 4 * h)).astype(np.float32)
        self.b = np.zeros((4 * h,), dtype=np.float32)
        self.Wy = self.rng.uniform(-limit, limit, size=(h, 1)).astype(np.float32)
        self.by = np.zeros((1,), dtype=np.float32)
        self._adam_m = [np.zeros_like(p) for p in self.parameters]
        self._adam_v = [np.zeros_like(p) for p in self.parameters]
        self._adam_t = 0

    @property
    def parameters(self) -> list[np.ndarray]:
        return [self.W, self.b, self.Wy, self.by]

    def _forward(self, X: np.ndarray, training: bool) -> tuple[np.ndarray, dict[str, object]]:
        batch, steps, _ = X.shape
        h_units = self.config.hidden_units
        h = np.zeros((batch, h_units), dtype=np.float32)
        c = np.zeros((batch, h_units), dtype=np.float32)
        caches = []
        for t in range(steps):
            x_t = X[:, t, :]
            concat = np.concatenate([x_t, h], axis=1)
            gates = concat @ self.W + self.b
            i = _sigmoid(gates[:, :h_units])
            f = _sigmoid(gates[:, h_units : 2 * h_units])
            g = np.tanh(gates[:, 2 * h_units : 3 * h_units])
            o = _sigmoid(gates[:, 3 * h_units :])
            c_prev = c
            h_prev = h
            c = f * c + i * g
            tanh_c = np.tanh(c)
            h = o * tanh_c
            caches.append((concat, i, f, g, o, c, c_prev, h_prev, tanh_c))

        keep_prob = 1.0 - float(np.clip(self.config.dropout, 0.0, 0.8))
        mask = None
        h_out = h
        if training and keep_prob < 1.0:
            mask = (self.rng.random(h.shape) < keep_prob).astype(np.float32) / keep_prob
            h_out = h * mask
        y_pred = h_out @ self.Wy + self.by
        return y_pred, {"caches": caches, "h_final": h, "mask": mask, "h_out": h_out}

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        preds = []
        for start in range(0, len(X), batch_size):
            pred, _ = self._forward(X[start : start + batch_size], training=False)
            preds.append(pred)
        return np.vstack(preds)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
        patience: int = 4,
    ) -> list[float]:
        best_val = float("inf")
        stale = 0
        history = []
        for _ in range(epochs):
            order = self.rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = order[start : start + batch_size]
                self._train_batch(X[idx], y[idx])
            val_pred = self.predict(X_val)
            val_loss = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))
            history.append(val_loss)
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        return history

    def _train_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        pred, cache = self._forward(X, training=True)
        batch = max(len(X), 1)
        dy = (2.0 / batch) * (pred - y)
        h_out = cache["h_out"]
        mask = cache["mask"]
        grads = [np.zeros_like(p) for p in self.parameters]
        grads[2][...] = h_out.T @ dy
        grads[3][...] = np.sum(dy, axis=0)
        dh = dy @ self.Wy.T
        if mask is not None:
            dh *= mask
        dc_next = np.zeros_like(dh)

        for concat, i, f, g, o, c, c_prev, _h_prev, tanh_c in reversed(cache["caches"]):
            do = dh * tanh_c
            dc = dh * o * (1.0 - tanh_c * tanh_c) + dc_next
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_next = dc * f

            di_in = di * i * (1.0 - i)
            df_in = df * f * (1.0 - f)
            dg_in = dg * (1.0 - g * g)
            do_in = do * o * (1.0 - o)
            d_gates = np.concatenate([di_in, df_in, dg_in, do_in], axis=1)
            grads[0][...] += concat.T @ d_gates
            grads[1][...] += np.sum(d_gates, axis=0)
            d_concat = d_gates @ self.W.T
            dh = d_concat[:, self.config.input_size :]

        for grad in grads:
            np.clip(grad, -1.0, 1.0, out=grad)
        self._adam_update(grads)

    def _adam_update(self, grads: list[np.ndarray]) -> None:
        self._adam_t += 1
        lr = self.config.learning_rate
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for p, g, m, v in zip(self.parameters, grads, self._adam_m, self._adam_v):
            m *= beta1
            m += (1.0 - beta1) * g
            v *= beta2
            v += (1.0 - beta2) * (g * g)
            m_hat = m / (1.0 - beta1**self._adam_t)
            v_hat = v / (1.0 - beta2**self._adam_t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)
