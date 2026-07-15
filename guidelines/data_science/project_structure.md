---
title: Estructura de proyectos de Ciencia de Datos
domain: data_science
tags: [estructura, cookiecutter, organizacion]
---

# Estructura de proyecto de Ciencia de Datos

Basado en convenciones tipo *Cookiecutter Data Science*. Una estructura clara
hace el trabajo reproducible y colaborable.

## Layout recomendado
```
proyecto/
├── data/
│   ├── raw/          # datos originales, inmutables (nunca se editan a mano)
│   ├── interim/      # datos intermedios transformados
│   └── processed/    # datasets finales listos para modelar
├── notebooks/        # exploración (numerados: 01-eda, 02-features…)
├── src/ (o pkg/)     # código reutilizable importable
│   ├── data/         # carga y validación
│   ├── features/     # ingeniería de variables
│   ├── models/       # entrenamiento y evaluación
│   └── viz/          # gráficos
├── models/           # modelos serializados (artefactos)
├── reports/          # reportes y figuras generadas
├── tests/
├── pyproject.toml / requirements.txt
└── README.md
```

## Principios
- **`data/raw` es sagrado**: inmutable; toda transformación se deriva por código.
- **Código en `src/`, no en notebooks**: los notebooks importan funciones de `src/`.
- **Separar configuración de código** (paths, hiperparámetros en config/env/yaml).
- **Un flujo reproducible**: de `raw` a `processed` con scripts, no pasos manuales.
- **README** con cómo obtener datos, instalar, entrenar y evaluar.

## Señales en revisión
- Rutas absolutas hardcodeadas a la máquina de alguien.
- Datos crudos modificados in-place o commiteados al repo.
- Toda la lógica dentro de un notebook gigante sin código reutilizable.
- Falta de separación entre EDA, features, modelado y evaluación.
