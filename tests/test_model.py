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
