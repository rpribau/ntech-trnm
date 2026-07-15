---
title: Design Patterns (catálogo GoF) y cuándo usarlos
domain: software_dev
tags: [design-patterns, gof, arquitectura]
---

# Design Patterns

Catálogo de referencia (Gang of Four) para **reconocer** y **evaluar** patrones en
código. En revisión: identifica si un patrón está bien aplicado o si se está
usando donde no aporta (over-engineering).

## Creacionales
- **Factory Method / Abstract Factory**: crear objetos sin acoplar al tipo concreto.
  *Señal*: función/clase que decide qué implementación instanciar según config.
- **Builder**: construir objetos complejos por pasos. *Señal*: constructores con
  muchísimos parámetros opcionales.
- **Singleton**: una única instancia global. **Úsalo con cuidado**: dificulta tests
  y crea estado global. Prefiere inyección de dependencias.
- **Prototype**: clonar objetos existentes.

## Estructurales
- **Adapter**: envuelve una interfaz para que encaje con otra (fronteras con libs).
- **Facade**: interfaz simple sobre un subsistema complejo.
- **Decorator**: añade comportamiento envolviendo, sin modificar la clase.
- **Composite**: trata objetos individuales y composiciones de forma uniforme (árboles).
- **Proxy**: sustituto que controla acceso (lazy, cache, permisos).
- **Bridge / Flyweight**: separar abstracción de implementación / compartir estado.

## De comportamiento
- **Strategy**: familia de algoritmos intercambiables (alternativa a `if` por tipo).
- **Observer**: notificación a suscriptores ante cambios (eventos).
- **Command**: encapsula una acción como objeto (undo, colas, logs).
- **State**: comportamiento según estado interno, sin `if` gigantes.
- **Template Method**: esqueleto de algoritmo con pasos redefinibles.
- **Iterator / Chain of Responsibility / Mediator / Visitor / Memento**: recorrer,
  encadenar handlers, centralizar comunicación, operar sobre estructuras, snapshots.

## Patrones frecuentes fuera de GoF
- **Repository**: abstrae el acceso a datos.
- **Dependency Injection**: proveer dependencias desde fuera.
- **Supervisor / Orchestrator** (agentes): un router coordina sub-agentes
  (patrón usado en este proyecto: ver el grafo LangGraph).
- **Pipeline**: etapas encadenadas (ingesta → chunking → embed → index).

## Criterio de revisión
- ¿El patrón **resuelve un problema real** aquí o es complejidad gratuita?
- ¿Está **completo y correcto** (p. ej. Strategy con interfaz común clara)?
- ¿Un patrón más simple bastaría? Evita el "patrón por el patrón".
- Documenta el patrón usado cerca del código para el siguiente lector.
