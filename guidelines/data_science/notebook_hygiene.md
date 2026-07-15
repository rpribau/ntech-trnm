---
title: Higiene de notebooks
domain: data_science
tags: [notebooks, jupyter, eda]
---

# Higiene de notebooks

Los notebooks son geniales para explorar, peligrosos para producción.

## Buenas prácticas
- **Corre de arriba a abajo** sin errores (Restart & Run All). Evita estado oculto
  por ejecución fuera de orden.
- **Numera** los notebooks por etapa (`01-eda.ipynb`, `02-features.ipynb`).
- Extrae la lógica reutilizable a `src/` e **impórtala**; el notebook orquesta.
- Limpia **outputs pesados** antes de commitear (o usa `nbstripout`/Jupytext).
- Sin credenciales ni rutas absolutas locales; usa config.
- Documenta con celdas markdown el objetivo y las conclusiones de cada sección.

## Anti-patrones
- Notebook monolítico de miles de celdas que lo hace todo.
- Copiar-pegar bloques entre notebooks en vez de una función compartida.
- Dejar celdas de prueba/basura y variables globales confusas.
- Depender del orden manual de ejecución para obtener el resultado.

## De notebook a producción
- Cuando algo funciona, **muévelo a un módulo** con tests.
- Considera `papermill`/Jupytext para parametrizar y versionar como texto.

## Señales en revisión
- Notebooks que no corren limpio de principio a fin.
- Lógica crítica atrapada solo en un notebook, sin tests ni reutilización.
- Outputs enormes o datos sensibles versionados en el `.ipynb`.
