# Detección de fraude en apertura de cuentas (BAF)

> English summary: fraud model on the Bank Account Fraud (NeurIPS 2022) dataset, with a temporal
> split, bank-style validation (Gini, KS, PSI), fairness checks and a FastAPI scoring service.

## Problema
Detectar solicitudes fraudulentas de apertura de cuenta. El fraude es raro (cerca del 1% de las
solicitudes) y cada falsa alarma afecta a un cliente legítimo, así que el modelo se evalúa por el
fraude que detecta (recall) con una tasa de falsos positivos (FPR) limitada.

## Datos
Bank Account Fraud (Feedzai, NeurIPS 2022), un conjunto sintético generado a partir de datos
bancarios. No se incluye en este repo: descárgalo de la fuente oficial y coloca `Base.csv` en
`data/raw/`.

## Método
- **Partición temporal:** entrenamiento meses 0–4, validación mes 5, prueba meses 6–7.
- **Decisiones** (variables, modelo, calibrador y umbrales) tomadas sin usar la prueba.
- **Comparación:** regresión logística con splines, LightGBM y XGBoost quedaron empatados
  estadísticamente; Random Forest fue el peor. Modelo final: LightGBM.
- **Variables descartadas:** `velocity_4w` (deriva monótona, actúa como reloj) y `customer_age`
  (atributo protegido). En ambos casos el Gini no cambió.
- **Calibración** de Platt y umbrales por FPR objetivo, ajustados en validación.
- **Revisión:** PSI para la deriva, SHAP para explicabilidad y análisis de equidad.

## Resultados
| Métrica | Validación (mes 5) | Prueba (meses 6–7) |
|---|---|---|
| Gini | 0.788 | 0.784 |
| KS | 0.627 | 0.628 |
| Recall @ 5% FPR | 0.534 | 0.541 |
| Recall @ 1% FPR | 0.280 | 0.252 |

Umbrales fijados en validación y aplicados a la prueba:

| FPR objetivo | FPR observado | Recall |
|---|---|---|
| 5% | 5.5% | 56.4% |
| 1% | 1.28% | 29.1% |

El notebook de exploración muestra cifras ligeramente distintas; las oficiales son las del
pipeline (`fraud.train`).

## Limitaciones
- **Calibración:** la probabilidad sin calibrar subestima el fraude. Platt la corrige en el mes 6,
  pero en el mes 7, con el fraude en 1.47%, vuelve a quedar corta. En producción habría que
  recalibrar periódicamente.
- **Deriva:** los umbrales se desvían de su objetivo (10–28%). `credit_risk_score` deriva con
  fuerza (PSI > 0.25), aunque pesa poco en el modelo.
- **Recall al 1% de FPR:** menor en prueba que en validación.
- **Equidad:** quitar la edad redujo la brecha de FPR entre 20 y 60 años de cerca de 10 veces a
  cerca de 5, pero no la eliminó (variables correlacionadas y distinta tasa base).
- **Grupos pequeños** (edades extremas; situación laboral CD, CE, CF, CG): no permiten
  conclusiones fiables.

## Cómo reproducirlo
Desde la raíz del repo:

```bash
conda env create -f environment.yml
conda activate fraude-baf
pytest -q

python -m fraud.train              # entrena, valida y guarda artifacts/
python -m fraud.train --eval-test  # además reporta la prueba
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## API
Requiere haber entrenado antes.

```bash
uvicorn fraud.api:app
```

Documentación interactiva en http://127.0.0.1:8000/docs, con `GET /health` y `POST /predict`.

## Estructura
- `src/fraud/`: datos, métricas, modelo, entrenamiento y API
- `tests/`: pruebas
- `notebooks/`: análisis exploratorio
- `reports/figures/`: figuras

## Pendiente
- Imagen Docker
- Figuras de SHAP, matriz de confusión y equidad