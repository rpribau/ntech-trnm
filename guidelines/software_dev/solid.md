---
title: Principios SOLID
domain: software_dev
tags: [solid, oop, diseño, arquitectura]
---

# SOLID

Cinco principios de diseño orientado a objetos para código flexible y desacoplado.

## S — Single Responsibility Principle (SRP)
Una clase/módulo debe tener **una sola razón para cambiar**. Si mezcla lógica de
negocio, acceso a datos y presentación, sepárala.
- *Olor*: clase "God object" que lo hace todo.

## O — Open/Closed Principle (OCP)
Abierto a **extensión**, cerrado a **modificación**. Agregar un caso nuevo no
debería obligarte a editar código existente y probado.
- *Técnica*: polimorfismo/estrategias en vez de cadenas `if/elif` por tipo.

## L — Liskov Substitution Principle (LSP)
Un subtipo debe poder usarse donde se espera el tipo base **sin romper**
expectativas (precondiciones no más fuertes, postcondiciones no más débiles).
- *Olor*: subclase que lanza `NotImplementedError` en métodos heredados.

## I — Interface Segregation Principle (ISP)
Mejor varias interfaces pequeñas y específicas que una grande. Un cliente no
debería depender de métodos que no usa.

## D — Dependency Inversion Principle (DIP)
Los módulos de alto nivel no dependen de los de bajo nivel; ambos dependen de
**abstracciones**. Inyecta dependencias (constructor/parámetros) en vez de crearlas
dentro (ver *factories over singletons*).
- *Beneficio*: testeable (mocks), intercambiable (p. ej. cambiar backend LLM).

## Cómo aplicarlo en revisión
- ¿La clase tiene más de una responsabilidad clara? → SRP.
- ¿Agregar una variante obliga a tocar un `switch` gigante? → OCP.
- ¿Hay dependencias concretas instanciadas dentro que impiden testear? → DIP.
- No sobre-diseñes: aplica SOLID donde el cambio es probable, no en todo.
