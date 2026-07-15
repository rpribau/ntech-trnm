---
title: Checklist de code review
domain: software_dev
tags: [code-review, checklist, calidad]
---

# Checklist de code review

Lista de verificación para revisar un repo o un cambio. Marca hallazgos con
severidad: **crítico / mayor / menor / sugerencia**.

## Correctitud
- [ ] ¿El código hace lo que dice? ¿Casos borde y errores contemplados?
- [ ] ¿Manejo de excepciones adecuado (sin tragar errores)?
- [ ] ¿Condiciones de carrera / concurrencia si aplica?

## Diseño y arquitectura
- [ ] ¿Responsabilidades claras (SRP)? ¿Acoplamiento bajo, cohesión alta?
- [ ] ¿Se respetan las capas y la dirección de dependencias?
- [ ] ¿Design patterns bien aplicados (no over-engineering)?
- [ ] ¿Duplicación evitable (DRY)? ¿Complejidad justificada (KISS/YAGNI)?

## Legibilidad
- [ ] ¿Nombres claros? ¿Funciones pequeñas y con un propósito?
- [ ] ¿Sin números/strings mágicos ni código muerto?
- [ ] ¿Comentarios que explican el porqué donde hace falta?

## Testing
- [ ] ¿Hay tests para la lógica nueva/no trivial? ¿Cubren errores?
- [ ] ¿Los tests son deterministas e independientes?

## Seguridad
- [ ] ¿Secretos fuera del código (config/env)? ¿Nada hardcodeado?
- [ ] ¿Validación de entradas? ¿Consultas parametrizadas (no inyección)?
- [ ] ¿Dependencias sin vulnerabilidades conocidas conocidas?

## Rendimiento
- [ ] ¿Complejidad razonable? ¿Sin N+1 ni trabajo repetido evitable?
- [ ] ¿Recursos liberados (context managers, conexiones)?

## Documentación y mantenibilidad
- [ ] ¿README/docstrings suficientes para entender y correr el proyecto?
- [ ] ¿Configuración externalizada? ¿Reproducibilidad (deps fijadas)?

## Salida esperada de la revisión
Para cada hallazgo: **archivo:línea**, categoría, severidad, por qué importa,
guía aplicable citada, y una recomendación accionable.
