#!/usr/bin/env bash
# Agrega el pipeline de entrenamiento reproducible (model.py, train.py, tests).
# Correr desde la RAÍZ del repo: bash add_training.sh
# Solo CREA archivos nuevos (no pisa los existentes) y agrega líneas al .gitignore.
set -euo pipefail

if [ ! -d src/fraud ] || [ ! -f src/fraud/data.py ]; then
  echo "Corre este script desde la raíz de fraude-bancario-baf (no encuentro src/fraud/data.py)." >&2
  exit 1
fi

for f in src/fraud/model.py src/fraud/train.py tests/test_model.py; do
  if [ -e "$f" ]; then
    echo "Ya existe $f; no lo piso. Bórralo o renómbralo si quieres regenerarlo." >&2
    exit 1
  fi
done

mkdir -p tests artifacts

# ---------------------------------------------------------------- model.py
cat > src/fraud/model.py <<'EOF'
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
EOF

# ---------------------------------------------------------------- train.py
cat > src/fraud/train.py <<'EOF'
"""Entrena el modelo final, lo registra en MLflow y guarda los artefactos.

Uso (desde la raíz del repo, con el entorno activado):
    python -m fraud.train                # entrena y mide en validación
    python -m fraud.train --eval-test    # además mide en los meses de prueba (hazlo con cuidado)
"""

from __future__ import annotations

import argparse
import subprocess

import mlflow

from fraud.data import MONTH, TARGET, load_baf
from fraud.metrics import gini, ks_statistic, recall_at_fpr
from fraud.model import (
    DROPPED,
    LGBM_PARAMS,
    FraudScorer,
    apply_platt,
    category_levels,
    feature_names,
    fit_platt,
    fpr_thresholds,
    prepare,
    train_model,
)

TRAIN_MONTHS = list(range(5))
VALID_MONTHS = [5]
TEST_MONTHS = [6, 7]


def _metrics(y, s, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_gini": float(gini(y, s)),
        f"{prefix}_ks": float(ks_statistic(y, s)),
        f"{prefix}_recall_fpr5": float(recall_at_fpr(y, s, 0.05)),
        f"{prefix}_recall_fpr1": float(recall_at_fpr(y, s, 0.01)),
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def main(data_path: str, out_dir: str, eval_test: bool) -> None:
    df = load_baf(data_path)
    train = df[df[MONTH].isin(TRAIN_MONTHS)]
    valid = df[df[MONTH].isin(VALID_MONTHS)]

    features = feature_names(df.columns)
    levels = category_levels(train, features)
    X_tr, y_tr = prepare(train, features, levels), train[TARGET].to_numpy()
    X_va, y_va = prepare(valid, features, levels), valid[TARGET].to_numpy()
    print(f"{len(features)} variables | train {len(train):,} filas | valid {len(valid):,} filas")

    model = train_model(X_tr, y_tr)
    s_va = model.predict_proba(X_va)[:, 1]

    # Todo lo que se decide, se decide SOLO con validación
    thresholds = fpr_thresholds(s_va, y_va)
    platt = fit_platt(s_va, y_va)
    metrics = _metrics(y_va, s_va, "valid")
    metrics["valid_pred_media_cruda"] = float(s_va.mean())
    metrics["valid_pred_media_calibrada"] = float(apply_platt(s_va, platt).mean())
    metrics["valid_fraude_real"] = float(y_va.mean())

    meta = {
        "features": features,
        "levels": levels,
        "dropped": list(DROPPED),
        "thresholds": thresholds,
        "platt": platt,
        "metrics_valid": {k: v for k, v in metrics.items() if k.startswith("valid_")},
    }
    scorer = FraudScorer(model, meta)

    if eval_test:
        test = df[df[MONTH].isin(TEST_MONTHS)]
        y_te = test[TARGET].to_numpy()
        s_te = scorer.score(test)["score"].to_numpy()
        metrics.update(_metrics(y_te, s_te, "test"))
        for key, thr in thresholds.items():
            flag = s_te >= thr
            metrics[f"test_fpr_obs_{key}"] = float(flag[y_te == 0].mean())
            metrics[f"test_recall_{key}"] = float(flag[y_te == 1].mean())
        meta["metrics_test"] = {k: v for k, v in metrics.items() if k.startswith("test_")}

    scorer.save(out_dir)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraude-baf")
    with mlflow.start_run(run_name="lightgbm-final"):
        mlflow.set_tags({"git_commit": _git_commit(), "eval_test": str(eval_test)})
        mlflow.log_params({**LGBM_PARAMS, "n_features": len(features),
                           "dropped": ",".join(DROPPED)})
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(out_dir)

    for name, value in metrics.items():
        print(f"{name:32s} {value:.4f}")
    print(f"\nArtefactos en {out_dir}/ | ver experimentos: "
          "mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/raw/Base.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--eval-test", action="store_true",
                    help="mide también en los meses 6-7 (úsalo solo para reportar)")
    args = ap.parse_args()
    main(args.data, args.out, args.eval_test)
EOF

# ---------------------------------------------------------------- tests
cat > tests/test_model.py <<'EOF'
import numpy as np
import pandas as pd

from fraud.model import (
    FraudScorer,
    apply_platt,
    category_levels,
    feature_names,
    fit_platt,
    fpr_thresholds,
    prepare,
    train_model,
)

SMALL = {"n_estimators": 20, "min_child_samples": 5, "random_state": 0, "verbose": -1}


def _df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "fraud_bool": rng.binomial(1, 0.1, n),
            "month": rng.integers(0, 8, n),
            "income": rng.random(n),
            "customer_age": rng.choice([20, 30, 40, 50], n),
            "velocity_4w": rng.normal(5000, 500, n),
            "velocity_6h": rng.normal(5000, 500, n),
            "prev_address_months_count": rng.integers(-1, 100, n),
            "current_address_months_count": rng.integers(-1, 100, n),
            "bank_months_count": rng.integers(-1, 30, n),
            "session_length_in_minutes": rng.random(n) * 10,
            "device_distinct_emails_8w": rng.integers(-1, 3, n),
            "device_os": rng.choice(["linux", "windows", "other"], n),
        }
    )
    df.loc[0, "velocity_6h"] = -5.0
    df["fraud_bool"] = df["fraud_bool"] | (df["income"] > 0.95).astype(int)
    return df


def test_feature_names_excludes_target_month_and_dropped():
    feats = feature_names(_df().columns)
    for banned in ("fraud_bool", "month", "velocity_4w", "customer_age"):
        assert banned not in feats
    assert "income" in feats and "device_os" in feats


def test_prepare_cleans_and_fixes_categories():
    df = _df()
    feats = feature_names(df.columns)
    levels = category_levels(df, feats)
    assert list(levels) == ["device_os"]

    X = prepare(df, feats, levels)
    assert X["prev_address_months_count"].isna().any()
    assert X.loc[0, "velocity_6h"] != X.loc[0, "velocity_6h"]  # NaN
    assert (X["prev_address_months_count"].dropna() >= 0).all()
    assert str(X["device_os"].dtype) == "category"

    nuevo = df.head(3).copy()
    nuevo["device_os"] = "temple_os"
    X2 = prepare(nuevo, feats, levels)
    assert X2["device_os"].isna().all()
    assert list(X2["device_os"].cat.categories) == levels["device_os"]


def test_thresholds_hit_target_fpr():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.05, 50_000)
    s = rng.random(50_000) + 0.5 * y
    thr = fpr_thresholds(s, y, fprs=(0.05, 0.01))
    for key, t in thr.items():
        obs = (s[y == 0] >= t).mean()
        assert abs(obs - float(key)) < 0.002


def test_platt_matches_base_rate_and_is_monotone():
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.02, 20_000)
    s = np.clip(0.01 + 0.03 * y + rng.random(20_000) * 0.02, 1e-4, 0.99)
    p = fit_platt(s, y)
    cal = apply_platt(s, p)
    assert abs(cal.mean() - y.mean()) < 0.002
    order = np.argsort(s)
    assert (np.diff(cal[order]) >= -1e-12).all()


def test_scorer_roundtrip(tmp_path):
    df = _df()
    feats = feature_names(df.columns)
    levels = category_levels(df, feats)
    X, y = prepare(df, feats, levels), df["fraud_bool"].to_numpy()
    model = train_model(X, y, SMALL)
    s = model.predict_proba(X)[:, 1]
    meta = {
        "features": feats,
        "levels": levels,
        "thresholds": fpr_thresholds(s, y),
        "platt": fit_platt(s, y),
    }
    FraudScorer(model, meta).save(tmp_path)
    out = FraudScorer.load(tmp_path).score(df)

    assert list(out.columns) == ["score", "prob_calibrada", "alerta_fpr_0.05", "alerta_fpr_0.01"]
    assert np.allclose(out["score"], s)
    assert out["alerta_fpr_0.01"].sum() <= out["alerta_fpr_0.05"].sum()
EOF

# ---------------------------------------------------------------- .gitignore
touch .gitignore
for pat in "mlflow.db" "mlruns/" "mlartifacts/" "artifacts/"; do
  grep -qxF "$pat" .gitignore || echo "$pat" >> .gitignore
done

echo
echo "Listo. Archivos creados: src/fraud/model.py, src/fraud/train.py, tests/test_model.py"
echo "Siguiente:"
echo "  1) Agrega 'mlflow' (y 'lightgbm' si no está) a la sección pip: de environment.yml e instálalos."
echo "  2) Pega los hiperparámetros de tu lgbm() en LGBM_PARAMS (src/fraud/model.py)."
echo "  3) pytest -q && ruff check ."
echo "  4) python -m fraud.train"
