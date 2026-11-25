"""API de puntuación de fraude.

Uso (desde la raíz del repo):
    uvicorn fraud.api:app --reload
"""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from fraud.model import FraudScorer


def to_frame(row: dict, features: list[str], levels: dict[str, list[str]]) -> pd.DataFrame:
    """Valida una fila cruda y la convierte en un DataFrame con los tipos correctos."""
    errors = [f"falta la variable '{c}'" for c in features if c not in row]
    for col in features:
        value = row.get(col)
        if value is None:
            continue
        if col in levels:
            if not isinstance(value, str):
                errors.append(f"'{col}' debe ser texto o null")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            errors.append(f"'{col}' debe ser un número o null")
    if errors:
        raise ValueError("; ".join(errors))

    df = pd.DataFrame([{c: row[c] for c in features}])
    # Un null en una sola fila dejaría la columna como 'object' y LightGBM fallaría
    numeric = [c for c in features if c not in levels]
    return df.astype({c: "float64" for c in numeric})


def create_app(scorer: FraudScorer | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # El modelo se carga una sola vez, al arrancar
        app.state.scorer = scorer or FraudScorer.load(os.getenv("ARTIFACTS_DIR", "artifacts"))
        yield

    app = FastAPI(title="Detección de fraude (BAF)", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        return {"status": "ok", "n_features": len(request.app.state.scorer.meta["features"])}

    @app.post("/predict")
    def predict(row: dict, request: Request) -> dict:
        model = request.app.state.scorer
        features, levels = model.meta["features"], model.meta["levels"]
        try:
            df = to_frame(row, features, levels)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result = model.score(df).to_dict(orient="records")[0]
        # Una categoría nueva se puntúa como faltante: se avisa en vez de ocultarlo
        result["categorias_desconocidas"] = [
            c for c, cats in levels.items() if row[c] is not None and row[c] not in cats
        ]
        return result

    return app


app = create_app()
