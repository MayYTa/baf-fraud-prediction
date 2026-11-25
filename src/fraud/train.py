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
