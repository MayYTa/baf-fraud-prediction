import numpy as np

from fraud.metrics import gini, ks_statistic, psi


def test_modelo_perfecto():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks_statistic(y, s) == 1.0
    assert gini(y, s) == 1.0


def test_psi_misma_distribucion_es_cercano_a_cero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, rng.normal(size=5000)) < 0.05


def test_psi_detecta_cambio():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    movida = rng.normal(1.5, 1, 5000)
    assert psi(base, movida) > 0.25
