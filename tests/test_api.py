import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud.api import create_app
from fraud.data import MISSING_MINUS_ONE
from fraud.model import FraudScorer, category_levels, prepare, train_model

FEATURES = [*MISSING_MINUS_ONE, "velocity_6h", "payment_type"]


@pytest.fixture(scope="module")
def client():
    """Modelo diminuto con datos sintéticos: no necesita el dataset ni artifacts/."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.integers(-1, 50, 200) for c in FEATURES[:-1]})
    df["payment_type"] = rng.choice(["AA", "AB"], 200)
    levels = category_levels(df, FEATURES)
    model = train_model(
        prepare(df, FEATURES, levels),
        rng.integers(0, 2, 200),
        {"n_estimators": 5, "verbose": -1},
    )
    meta = {
        "features": FEATURES,
        "levels": levels,
        "platt": {"a": 1.0, "b": 0.0},
        "thresholds": {"0.05": 0.9, "0.01": 0.95},
    }
    with TestClient(create_app(FraudScorer(model, meta))) as c:
        yield c


def row(**changes):
    return {**{c: 1 for c in FEATURES[:-1]}, "payment_type": "AA", **changes}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "n_features": len(FEATURES)}


def test_predict(client):
    body = client.post("/predict", json=row()).json()
    assert 0 <= body["score"] <= 1
    assert 0 <= body["prob_calibrada"] <= 1
    assert body["categorias_desconocidas"] == []


def test_predict_accepts_nulls(client):
    response = client.post("/predict", json=row(velocity_6h=None, payment_type=None))
    assert response.status_code == 200


def test_unknown_category_is_reported(client):
    body = client.post("/predict", json=row(payment_type="ZZ")).json()
    assert body["categorias_desconocidas"] == ["payment_type"]


def test_missing_feature_is_rejected(client):
    bad = row()
    del bad["velocity_6h"]
    assert client.post("/predict", json=bad).status_code == 422


def test_wrong_type_is_rejected(client):
    assert client.post("/predict", json=row(velocity_6h="mucho")).status_code == 422
