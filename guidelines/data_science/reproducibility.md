---
title: Reproducibilidad
domain: data_science
tags: [reproducibilidad, semillas, entorno, pipelines]
---

# Reproducibilidad

Un resultado que no se puede reproducir no es un resultado.

## Entorno
- **Fija dependencias** con versiones exactas (`requirements.txt`/`pyproject`/lockfile).
- Entorno aislado (venv/conda/uv/docker). Documenta versión de Python y del SO si importa.
- Contenedor (Docker) para experimentos críticos o despliegue.

## Determinismo
- Fija **semillas** (`random`, `numpy`, framework de ML) y documéntalo.
- Ojo con fuentes de no-determinismo: paralelismo, GPU, orden de datos, hashing.
- Registra la **versión de los datos** usados (hash/fecha/tag).

## Pipeline reproducible
- De datos crudos a resultado con **código ejecutable**, no pasos manuales.
- Parámetros e hiperparámetros en **config** versionada, no incrustados.
- Herramientas: `Makefile`/`invoke`, DVC, Snakemake, o pipelines de MLflow.

## Trazabilidad
- Registra para cada experimento: datos, código (commit), config, métricas y artefactos.
- Un experimento debe poder re-ejecutarse desde su registro.

## Señales en revisión
- "Funciona en mi máquina": deps no fijadas, rutas locales, pasos manuales.
- Resultados sin semilla ni versión de datos.
- Notebooks con estado oculto que no corren de arriba a abajo.
