"""Metricas de validacion tipo banco: KS, Gini y PSI."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def ks_statistic(y_true, y_score) -> float:
    """KS: maxima separacion entre las curvas de TPR y FPR."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def gini(y_true, y_score) -> float:
    """Gini = 2 * AUC - 1."""
    return float(2 * roc_auc_score(y_true, y_score) - 1)


def psi(expected, actual, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index entre dos muestras de una variable o score.

    Los cortes se calculan con cuantiles de `expected` (periodo de entrenamiento).
    Regla comun: < 0.10 estable, 0.10-0.25 vigilar, > 0.25 cambio importante.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=cuts)[0] / len(expected)
    a = np.histogram(actual, bins=cuts)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def recall_at_fpr(y_true, y_score, max_fpr: float = 0.05) -> float:
    """Fraude capturado (TPR) permitiendo como maximo `max_fpr` de falsos positivos."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(tpr[fpr <= max_fpr].max())