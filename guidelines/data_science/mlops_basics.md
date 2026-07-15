---
title: MLOps básico
domain: data_science
tags: [mlops, despliegue, monitoreo, ci-cd]
---

# MLOps básico

Llevar modelos de un notebook a un servicio confiable y mantenible.

## Ciclo de vida
1. **Datos**: ingestión, validación de esquema y calidad, versionado.
2. **Entrenamiento**: reproducible, con tracking de experimentos.
3. **Evaluación**: métricas + criterio de promoción claro.
4. **Despliegue**: empaquetado (contenedor), API o batch, versionado del modelo.
5. **Monitoreo**: latencia, errores, y **drift** de datos/predicciones.
6. **Reentrenamiento**: disparado por drift o degradación de métricas.

## Prácticas
- **CI/CD** también para ML: tests de datos, de código y del modelo.
- **Validación de datos** en la entrada (esquema, rangos, nulos) — falla temprano.
- **Registro de modelos** con metadatos y linaje (datos+código+params).
- **Reproducibilidad** de extremo a extremo (ver reproducibility.md).
- Separa **entrenamiento** de **inferencia**; la inferencia debe ser ligera y estable.
- Documenta el **contrato** del modelo (inputs, outputs, supuestos, límites).

## Monitoreo en producción
- Métricas técnicas (latencia, throughput, errores) y de negocio.
- **Data drift / concept drift**: la distribución cambia respecto a train.
- Alertas y un plan de rollback a la versión anterior del modelo.

## Señales en revisión
- Modelo servido sin versionado ni forma de reproducir su entrenamiento.
- Sin validación de entradas ni monitoreo/logging.
- Entrenamiento e inferencia acoplados en el mismo código frágil.
- Sin estrategia de reentrenamiento ni criterio de degradación.
