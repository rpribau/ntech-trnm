---
title: Anti-patterns y code smells
domain: software_dev
tags: [code-smells, anti-patterns, refactoring, deuda-tecnica]
---

# Anti-patterns y code smells

Catálogo para detectar deuda técnica en revisión.

## Principios violados (recordatorio)
- **DRY** (Don't Repeat Yourself): duplicación de lógica → extrae función/módulo.
- **KISS** (Keep It Simple): la solución más simple que funcione.
- **YAGNI** (You Aren't Gonna Need It): no construyas para un futuro hipotético.

## Code smells clásicos
- **God Object / God Function**: una clase/función que lo hace todo (viola SRP).
- **Long Method**: método largo con múltiples niveles de abstracción.
- **Long Parameter List**: muchos argumentos → objeto de parámetros / config.
- **Duplicated Code**: mismo bloque en varios lugares.
- **Feature Envy**: un método usa más datos de otra clase que de la suya.
- **Data Clumps**: grupos de variables que siempre viajan juntas → agrúpalas.
- **Primitive Obsession**: usar primitivos donde un tipo propio daría semántica.
- **Shotgun Surgery**: un cambio obliga a editar muchos archivos dispersos.
- **Magic Numbers/Strings**: literales sin nombre repartidos por el código.
- **Dead Code**: código inalcanzable o nunca usado.
- **Comentarios como muleta**: comentar código malo en vez de mejorarlo.

## Anti-patterns de arquitectura
- **Big Ball of Mud**: sin estructura ni capas claras.
- **Spaghetti Code**: flujo de control enredado, difícil de seguir.
- **Golden Hammer**: aplicar la misma herramienta/patrón a todo.
- **Copy-Paste Programming**: reutilizar por duplicación en vez de abstracción.
- **Hardcoding**: rutas, credenciales o parámetros incrustados (usa config).
- **Premature Optimization**: complejidad por rendimiento no medido.
- **Reinventing the Wheel**: reimplementar lo que una lib estándar ya resuelve.

## Smells específicos de Python
- Mutable default arguments (`def f(x=[])`).
- `except:` desnudo o `except Exception: pass` (traga errores).
- Comparaciones con `== None` en vez de `is None`.
- Imports con `from x import *`.
- No cerrar recursos (usa context managers `with`).

## En revisión
Para cada smell: nómbralo, di **por qué** importa (riesgo/costo) y propón un
refactor concreto y proporcional. Prioriza los que afectan correctitud y
mantenibilidad sobre los cosméticos.
