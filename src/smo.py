"""Implementación didáctica de SVM binaria mediante SMO.

El criterio de trabajo sigue la idea de *maximum violating pair*:
se eligen los índices que maximizan la brecha ``m - M`` de KKT y se
actualizan dos multiplicadores a la vez. La implementación permite comprobar
numéricamente la solución dual y las condiciones KKT en conjuntos pequeños.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


class SMOBinary:
    """SVM binaria con kernels lineal, polinomial y RBF."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: float = 0.1,
        degree: int = 3,
        coef0: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 20_000,
    ):
        if C <= 0:
            raise ValueError("C debe ser positivo")
        if kernel not in {"linear", "poly", "rbf"}:
            raise ValueError("kernel debe ser 'linear', 'poly' o 'rbf'")
        self.C = float(C)
        self.kernel = kernel
        self.gamma = float(gamma)
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def _kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        dot = X @ Y.T
        if self.kernel == "linear":
            return dot
        if self.kernel == "poly":
            return (self.gamma * dot + self.coef0) ** self.degree
        x2 = np.sum(X * X, axis=1)[:, None]
        y2 = np.sum(Y * Y, axis=1)[None, :]
        distances = np.maximum(x2 + y2 - 2.0 * dot, 0.0)
        return np.exp(-self.gamma * distances)

    def _working_sets(self, alpha: np.ndarray, y: np.ndarray):
        eps = 1e-10
        i_up = ((y == 1) & (alpha < self.C - eps)) | ((y == -1) & (alpha > eps))
        i_low = ((y == 1) & (alpha > eps)) | ((y == -1) & (alpha < self.C - eps))
        return i_up, i_low

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X debe ser bidimensional y coincidir con y")
        if set(np.unique(y)) != {-1.0, 1.0}:
            raise ValueError("y debe contener exactamente las etiquetas {-1, +1}")

        n = len(y)
        K = self._kernel(X, X)
        Q = np.outer(y, y) * K
        alpha = np.zeros(n, dtype=float)
        gradient = -np.ones(n, dtype=float)  # Q @ alpha - 1
        gap_history = []
        converged = False

        for iteration in range(1, self.max_iter + 1):
            i_up, i_low = self._working_sets(alpha, y)
            score = -y * gradient
            up_indices = np.flatnonzero(i_up)
            low_indices = np.flatnonzero(i_low)
            i = int(up_indices[np.argmax(score[up_indices])])
            m = float(score[i])

            # Probar primero los candidatos con menor puntuación. Si dos puntos
            # son idénticos, eta puede ser cero y se pasa al siguiente candidato.
            candidates = low_indices[np.argsort(score[low_indices])]
            updated = False
            for j_value in candidates:
                j = int(j_value)
                if j == i:
                    continue
                M = float(score[j])
                gap = m - M
                if gap <= self.tol:
                    converged = True
                    break

                eta = float(K[i, i] + K[j, j] - 2.0 * K[i, j])
                if eta <= 1e-14:
                    continue

                # Dirección factible: Δalpha_i=t*y_i, Δalpha_j=-t*y_j.
                lower_i, upper_i = (
                    (-alpha[i], self.C - alpha[i])
                    if y[i] == 1
                    else (alpha[i] - self.C, alpha[i])
                )
                lower_j, upper_j = (
                    (alpha[j] - self.C, alpha[j])
                    if y[j] == 1
                    else (-alpha[j], self.C - alpha[j])
                )
                lower = max(lower_i, lower_j)
                upper = min(upper_i, upper_j)
                if upper - lower <= 1e-14:
                    continue

                step = float(np.clip(gap / eta, lower, upper))
                if abs(step) <= 1e-14:
                    continue
                delta_i = step * y[i]
                delta_j = -step * y[j]
                alpha[i] += delta_i
                alpha[j] += delta_j
                gradient += Q[:, i] * delta_i + Q[:, j] * delta_j
                gap_history.append(gap)
                updated = True
                break

            if converged:
                break
            if not updated:
                break

        i_up, i_low = self._working_sets(alpha, y)
        score = -y * gradient
        m = float(np.max(score[i_up]))
        M = float(np.min(score[i_low]))
        free = (alpha > 1e-8) & (alpha < self.C - 1e-8)
        b = float(np.median(score[free])) if np.any(free) else (m + M) / 2.0

        z = alpha * y
        self.X_ = X
        self.y_ = y
        self.alpha_ = alpha
        self.b_ = b
        self.support_ = np.flatnonzero(alpha > 1e-8)
        self.n_iter_ = iteration
        self.converged_ = bool(converged or (m - M <= self.tol))
        self.m_ = m
        self.M_ = M
        self.kkt_gap_ = m - M
        self.equality_residual_ = float(abs(alpha @ y))
        self.dual_objective_ = float(alpha.sum() - 0.5 * z @ K @ z)
        self.gap_history_ = np.asarray(gap_history)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        sv = self.support_
        return self._kernel(X, self.X_[sv]) @ (self.alpha_[sv] * self.y_[sv]) + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)


class SMOOneVsRest:
    """Extensión OvR explícita de :class:`SMOBinary`."""

    def __init__(self, **binary_params):
        self.binary_params = binary_params

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_ = []
        for label in self.classes_:
            target = np.where(y == label, 1.0, -1.0)
            self.models_.append(SMOBinary(**self.binary_params).fit(X, target))
        return self

    def decision_function(self, X):
        return np.column_stack([model.decision_function(X) for model in self.models_])

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class SMOOneVsOne:
    """Extensión OvO explícita con voto y desempate por margen acumulado."""

    def __init__(self, **binary_params):
        self.binary_params = binary_params

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.class_to_index_ = {label: i for i, label in enumerate(self.classes_)}
        self.models_ = []
        for negative, positive in combinations(self.classes_, 2):
            mask = (y == negative) | (y == positive)
            target = np.where(y[mask] == positive, 1.0, -1.0)
            model = SMOBinary(**self.binary_params).fit(X[mask], target)
            self.models_.append((negative, positive, model))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        votes = np.zeros((len(X), len(self.classes_)), dtype=int)
        margins = np.zeros_like(votes, dtype=float)
        for negative, positive, model in self.models_:
            score = model.decision_function(X)
            i_neg = self.class_to_index_[negative]
            i_pos = self.class_to_index_[positive]
            votes[:, i_pos] += score >= 0
            votes[:, i_neg] += score < 0
            margins[:, i_pos] += score
            margins[:, i_neg] -= score
        # Un margen pequeño solo rompe empates; el número de votos domina.
        combined = votes + 1e-9 * margins
        return self.classes_[np.argmax(combined, axis=1)]
