# Deteccion de fraude en apertura de cuentas (BAF)

> Resumen en ingles: fraud model on the Bank Account Fraud (NeurIPS 2022) dataset,
> validated bank-style (KS, Gini, PSI), with fairness checks and a deployable API.

## Problema
(2-3 lineas: que se quiere detectar, por que importa, que costo tiene un falso positivo.)

## Datos
Bank Account Fraud (Feedzai, NeurIPS 2022). Descarga desde el repositorio oficial y
coloca `Base.csv` en `data/raw/`. No se incluye en este repo.

## Metodo
1. EDA y calidad de datos.
2. Particion **temporal** por mes (entrenamiento / validacion / prueba).
3. Modelo base (regresion logistica) vs LightGBM.
4. Validacion: KS, Gini, PSI del score y de las variables principales.
5. Explicabilidad (SHAP) y analisis de equidad por edad, ingreso y situacion laboral.
6. API con FastAPI + Docker + seguimiento con MLflow.

## Resultados
| Métrica | Validación (mes 5) | Prueba (meses 6–7) |
|---|---|---|
| Gini | 0.786 | 0.777 (IC95% 0.766–0.787) |
| KS | – | 0.623 |
| Recall @ 5% FPR | 0.531 | 0.540 (IC95% 0.522–0.558) |
| Recall @ 1% FPR | 0.271 | 0.239 |

Punto de operación (umbral fijado en validación): FPR objetivo 5% → observado 5.5%,
recall 55.5%, precisión 12.6%. Objetivo 1% → observado 1.2%, recall 26.7%, precisión 23.5%.

## Como reproducirlo
```bash
conda env create -f environment.yml
conda activate fraude-baf
pytest -q
```

## Estructura
`src/fraud/` codigo reutilizable, `notebooks/` analisis, `tests/` pruebas, `reports/` figuras.
