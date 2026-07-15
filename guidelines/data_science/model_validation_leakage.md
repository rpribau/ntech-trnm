---
title: Validación de modelos y fuga de datos (data leakage)
domain: data_science
tags: [validacion, data-leakage, metricas, overfitting]
---

# Validación de modelos y data leakage

## Data leakage (fuga de datos)
El error más caro en ML: información del target o del test se cuela en el entrenamiento.
- **Fit solo en train**: escaladores, imputadores, encoders, selección de features
  se ajustan con train y se aplican a val/test (usa `Pipeline`).
- **Sin features del futuro**: no uses variables que no existirían al momento de predecir.
- **Series de tiempo**: split temporal, no aleatorio; no mezcles pasado/futuro.
- **Datos agrupados**: si hay grupos (usuario, paciente), usa split por grupo para que
  el mismo grupo no esté en train y test.
- Cuidado con duplicados y con imputar/normalizar **antes** de separar.

## Validación
- Separa **train / validation / test**; el test se toca solo al final.
- **Cross-validation** apropiada al problema (estratificada, por grupo, temporal).
- Ajusta hiperparámetros con validación, **nunca** con el test.

## Métricas
- Elige métricas acordes al problema y al **desbalance** (no solo accuracy: usa
  precision/recall/F1, ROC-AUC/PR-AUC; para regresión MAE/RMSE/R²).
- Reporta una **baseline** y el intervalo/variabilidad, no un solo número.
- Verifica que la métrica refleje el objetivo de negocio.

## Overfitting / underfitting
- Compara train vs val: gran brecha → overfitting; ambos malos → underfitting.
- Regularización, más datos, o modelo más simple según el caso.

## Señales en revisión
- Escalado/encoding ajustado sobre todo el dataset antes del split.
- Uso del test para elegir modelo o hiperparámetros.
- Métrica única engañosa en datos desbalanceados.
- Split aleatorio en datos temporales o agrupados.
