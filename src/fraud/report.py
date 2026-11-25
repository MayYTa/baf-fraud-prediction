"""Genera las figuras del README a partir del modelo guardado en artifacts/.

Uso (desde la raíz del repo, después de `python -m fraud.train`):
    python -m fraud.report

Solo reporta: aquí no se ajusta ni se elige nada.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from fraud.data import MONTH, TARGET, load_baf
from fraud.model import FraudScorer, prepare

OUT = Path("reports/figures")


def _save(name: str) -> None:
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close("all")


def confusion(y: np.ndarray, scores: pd.DataFrame) -> None:
    keys = [c for c in scores.columns if c.startswith("alerta_fpr_")]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys, strict=True):
        cm = confusion_matrix(y, scores[key], labels=[False, True])
        ConfusionMatrixDisplay(cm, display_labels=["legítimo", "fraude"]).plot(
            ax=ax, values_format=",d", colorbar=False, cmap="Blues"
        )
        ax.set_title(f"Umbral FPR {float(key.removeprefix('alerta_fpr_')):.0%} (meses 6-7)")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
    fig.tight_layout()
    _save("matriz_confusion.png")


def explain(scorer: FraudScorer, df: pd.DataFrame) -> None:
    """SHAP en validación. LightGBM calcula los valores TreeSHAP con pred_contrib."""
    X = prepare(df, scorer.meta["features"], scorer.meta["levels"])
    values = scorer.model.booster_.predict(X, pred_contrib=True)[:, :-1]  # sin el valor base
    shown = X.copy()
    for col in scorer.meta["levels"]:
        shown[col] = shown[col].cat.codes.replace(-1, np.nan)
    for kind, name in (("bar", "shap_importancia.png"), ("dot", "shap_resumen.png")):
        shap.summary_plot(values, shown, plot_type=kind, max_display=15, show=False)
        _save(name)


def by_group(df: pd.DataFrame, flag: pd.Series, column: str) -> pd.DataFrame:
    """Falsas alarmas y recall por grupo, con el tamaño de cada uno."""
    rows = []
    for group, idx in df.groupby(column).groups.items():
        y, f = df.loc[idx, TARGET], flag.loc[idx]
        rows.append(
            {
                column: group,
                "n": len(idx),
                "fraudes": int(y.sum()),
                "fpr": f[y == 0].mean(),
                "recall": f[y == 1].mean(),
            }
        )
    return pd.DataFrame(rows)


def fairness(df: pd.DataFrame, scores: pd.DataFrame, column: str) -> None:
    table = by_group(df, scores["alerta_fpr_0.05"], column)
    table.to_csv(OUT / f"equidad_{column}.csv", index=False)
    ax = table.plot.bar(x=column, y="fpr", legend=False, figsize=(8, 4))
    ax.set_ylabel("Tasa de falsos positivos")
    ax.set_title(f"Falsas alarmas por {column} (umbral FPR 5%, meses 6-7)")
    for i, n in enumerate(table["n"]):
        ax.annotate(f"n={n:,}", (i, 0), xytext=(0, 3), textcoords="offset points",
                    rotation=90, ha="center", va="bottom", fontsize=7)
    _save(f"equidad_{column}.png")
    print(f"\n{table.to_string(index=False)}")


def main(data_path: str = "data/raw/Base.csv", artifacts: str = "artifacts") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scorer = FraudScorer.load(artifacts)
    df = load_baf(data_path)

    test = df[df[MONTH].isin([6, 7])]
    scores = scorer.score(test)
    confusion(test[TARGET].to_numpy(), scores)
    for column in ("customer_age", "employment_status"):
        fairness(test, scores, column)

    valid = df[df[MONTH] == 5]
    explain(scorer, valid.sample(min(len(valid), 5000), random_state=42))
    print(f"\nFiguras guardadas en {OUT}/")


if __name__ == "__main__":
    main()
