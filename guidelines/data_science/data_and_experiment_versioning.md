---
title: Versionado de datos y experimentos
domain: data_science
tags: [dvc, mlflow, tracking, datasets]
---

# Versionado de datos y experimentos

## Datos
- No commitees datasets grandes a git. Usa **DVC**, `git-lfs`, o almacenamiento
  externo (GCS/S3) con referencias versionadas.
- Guarda un **hash/manifiesto** del dataset usado en cada experimento.
- Documenta origen, licencia, esquema y fecha de extracción de cada fuente.
- Separa splits (train/val/test) de forma **fija y documentada** para comparabilidad.

## Experimentos
- Usa un tracker (**MLflow**, Weights & Biases, o registros propios) para guardar:
  parámetros, métricas, artefactos, código (commit) y entorno.
- Nombra y etiqueta experimentos de forma consistente.
- Compara modelos con las **mismas** métricas y el **mismo** split de test.

## Modelos
- Versiona los artefactos de modelo (registro de modelos) con metadatos:
  datos, métricas, fecha, autor.
- Define un criterio claro de "promoción" (qué métrica y umbral para pasar a prod).

## Señales en revisión
- Datasets o modelos binarios commiteados directo al repo.
- Experimentos sin registro de parámetros/métricas (no comparables).
- Splits generados al azar sin semilla ni persistencia (fugas entre corridas).
