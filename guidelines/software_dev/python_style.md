---
title: Estilo de Python (PEP 8 y buenas prácticas)
domain: software_dev
tags: [python, pep8, estilo, tipado]
---

# Estilo de Python

Resumen operativo de PEP 8 y prácticas idiomáticas ("pythonic").

## Formato (PEP 8)
- Indentación de 4 espacios; líneas ~ ≤ 88–100 chars (usa un formatter: ruff/black).
- `snake_case` para funciones/variables, `PascalCase` para clases,
  `UPPER_CASE` para constantes.
- Imports agrupados: stdlib, terceros, locales; sin `import *`.
- Dos líneas en blanco entre funciones/clases top-level.

## Idiomático (pythonic)
- Comprehensions claras en vez de bucles que solo construyen listas.
- Context managers (`with`) para archivos, conexiones, locks.
- `enumerate`, `zip`, desempaquetado; evita índices manuales.
- `pathlib.Path` en vez de manipular strings de rutas.
- f-strings para formateo.
- `is None` / `is not None` para comparar con `None`.
- Evita mutable default args; usa `None` y crea dentro.

## Tipado
- Usa **type hints** en firmas públicas (`def f(x: int) -> str:`).
- `from __future__ import annotations` para anotaciones diferidas.
- Considera `mypy`/`pyright` en CI para proyectos serios.

## Estructura y dependencias
- Layout de paquete claro; `pyproject.toml` como fuente de verdad.
- **Fija versiones** de dependencias para reproducibilidad.
- Entorno aislado (venv/poetry/uv). No dependas del Python del sistema.

## Docstrings
- Estilo consistente (Google/NumPy/reST). Documenta qué hace, args y retorno.
- Docstrings a nivel módulo/paquete para orientar al lector.

## Herramientas recomendadas
- **ruff** (lint + format), **mypy** (tipos), **pytest** (tests),
  **radon** (complejidad). Configúralas en `pyproject.toml`.
