---
title: Testing y calidad
domain: software_dev
tags: [testing, pytest, cobertura, calidad]
---

# Testing

## Pirámide de tests
- **Unitarios** (muchos, rápidos): funciones/clases aisladas.
- **Integración** (algunos): componentes juntos (DB, API, retrieval).
- **End-to-end** (pocos): flujo completo del usuario.
Evita la "copa de helado" invertida (todo E2E, lento y frágil).

## Buenas prácticas
- **FIRST**: Fast, Independent, Repeatable, Self-validating, Timely.
- Un test = un comportamiento; nombre descriptivo (`test_router_clasifica_sql`).
- Estructura **Arrange–Act–Assert**.
- Sin dependencias de orden ni de estado global entre tests.
- **Determinismo**: fija semillas, mockea tiempo/red/aleatoriedad.
- Testea también los **casos borde** y los errores, no solo el happy path.
- Los tests son código: aplícales las mismas reglas de limpieza.

## Qué mockear
- Servicios externos (LLM, GitHub, red), no la lógica que quieres probar.
- Para RAG/agentes: usa fixtures pequeñas y datos de ejemplo reproducibles.

## Cobertura
- La cobertura es una guía, no una meta ciega. 100% de líneas no implica corrección.
- Prioriza cubrir la lógica crítica y los caminos de error.

## Señales en revisión
- Código nuevo sin tests para la lógica no trivial.
- Tests que no afirman nada (sin `assert`) o que dependen de la red real.
- Falta de tests de regresión al corregir un bug (agrega uno que lo reproduzca).
