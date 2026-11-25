"""Preparación de datos, entrenamiento, calibración y umbrales del modelo final.

Todo lo que el modelo necesita para puntuar (features, categorías, calibrador, umbrales)
se guarda junto al modelo, de modo que la API aplique exactamente lo mismo que el notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from fraud.data import MONTH, TARGET, clean

# Variables que NO entran al modelo final (decisiones tomadas con datos de validación):
#  - velocity_4w: deriva monótona mes a mes (funciona como reloj); sin ella el Gini no cambia.
#  - customer_age: atributo protegido; sin ella el Gini no cambia y la brecha de FPR baja.
DROPPED = ("velocity_4w", "customer_age")

# TODO: reemplaza por los hiperparámetros de tu lgbm() del notebook para reproducir
# exactamente el Gini de validación (0.786).
LGBM_PARAMS: dict = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def feature_names(columns) -> list[str]:
    """Columnas que entran al modelo: todo menos target, mes y variables descartadas."""
    excluded = {TARGET, MONTH, *DROPPED}
    return [c for c in columns if c not in excluded]


def category_levels(df: pd.DataFrame, features: list[str]) -> dict[str, list[str]]:
    """Niveles de cada columna no numérica, fijados con los datos de entrenamiento."""
    return {
        c: sorted(df[c].dropna().astype(str).unique())
        for c in features
        if not pd.api.types.is_numeric_dtype(df[c])
    }


def prepare(df: pd.DataFrame, features: list[str], levels: dict[str, list[str]]) -> pd.DataFrame:
    """clean -> selección de features -> categorías con niveles fijos (valores nuevos -> NaN)."""
    out = clean(df)[features].copy()
    for col, cats in levels.items():
        valores = out[col].astype("object")
        out[col] = pd.Categorical(valores.where(valores.isin(cats)), categories=cats)
    return out


def train_model(X: pd.DataFrame, y, params: dict | None = None) -> LGBMClassifier:
    return LGBMClassifier(**(params or LGBM_PARAMS)).fit(X, y)


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(scores, y) -> dict[str, float]:
    """Calibrador de Platt (2 parámetros) sobre el logit del score."""
    lr = LogisticRegression(C=1e6).fit(_logit(scores).reshape(-1, 1), y)
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def apply_platt(scores, platt: dict[str, float]) -> np.ndarray:
    z = platt["a"] * _logit(scores) + platt["b"]
    return 1.0 / (1.0 + np.exp(-z))


def fpr_thresholds(scores, y, fprs=(0.05, 0.01)) -> dict[str, float]:
    """Umbral por FPR objetivo: cuantil (1 - fpr) de los scores de los casos legítimos."""
    neg = np.asarray(scores)[np.asarray(y) == 0]
    return {f"{f:g}": float(np.quantile(neg, 1 - f)) for f in fprs}


class FraudScorer:
    """Modelo + metadatos listos para puntuar filas crudas (como las del CSV original)."""

    def __init__(self, model: LGBMClassifier, meta: dict):
        self.model = model
        self.meta = meta

    @classmethod
    def load(cls, directory: str | Path) -> FraudScorer:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        return cls(joblib.load(directory / "model.joblib"), meta)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "model.joblib")
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False))

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        X = prepare(df, self.meta["features"], self.meta["levels"])
        raw = self.model.predict_proba(X)[:, 1]
        out = pd.DataFrame(index=df.index)
        out["score"] = raw
        out["prob_calibrada"] = apply_platt(raw, self.meta["platt"])
        for key, thr in self.meta["thresholds"].items():
            out[f"alerta_fpr_{key}"] = raw >= thr
        return out
