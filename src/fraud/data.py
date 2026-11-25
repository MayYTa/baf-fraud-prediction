"""Carga de BAF y particion temporal.

OJO: verifica los nombres de columnas contra la version del dataset que bajes
(se espera `fraud_bool` como objetivo y `month` como periodo).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "fraud_bool"
MONTH = "month"


def load_baf(path: str | Path = "data/raw/Base.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def temporal_split(
    df: pd.DataFrame,
    train_months: range = range(5),
    valid_months: range = range(5, 6),
    test_months: range = range(6, 8),
):
    """Entrena con los primeros meses y valida/prueba con los posteriores."""
    return (
        df[df[MONTH].isin(train_months)],
        df[df[MONTH].isin(valid_months)],
        df[df[MONTH].isin(test_months)],
    )


MISSING_MINUS_ONE = [
    "prev_address_months_count", "current_address_months_count", "bank_months_count",
    "session_length_in_minutes", "device_distinct_emails_8w",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte los faltantes codificados (-1) y el ruido de velocity_6h en NaN."""
    out = df.copy()
    out[MISSING_MINUS_ONE] = out[MISSING_MINUS_ONE].mask(out[MISSING_MINUS_ONE] == -1)
    out.loc[out["velocity_6h"] < 0, "velocity_6h"] = np.nan
    return out