---
title: Clean Code — principios de código limpio
domain: software_dev
tags: [clean-code, legibilidad, naming, funciones]
---

# Clean Code

Principios prácticos para escribir código legible y mantenible.

## Nombres
- **Nombres reveladores de intención**: `dias_desde_ultima_ejecucion` en vez de `d`.
- **Evita desinformación**: no llames `lista` a algo que no es una lista.
- **Pronunciables y buscables**: evita abreviaturas crípticas; constantes con nombre en vez de números mágicos.
- **Una palabra por concepto**: no mezcles `get`, `fetch`, `retrieve` para lo mismo.
- Clases = sustantivos; funciones/métodos = verbos.

## Funciones
- **Pequeñas y con un solo propósito** (Single Responsibility a nivel función).
- **Pocos parámetros** (idealmente ≤ 3). Muchos argumentos → agrupa en un objeto.
- **Sin efectos secundarios ocultos**: el nombre debe reflejar todo lo que hace.
- **Evita flags booleanos** que bifurcan el comportamiento; separa en dos funciones.
- **Un nivel de abstracción por función**: no mezcles orquestación con detalle fino.
- **Prefiere `return` temprano** (guard clauses) a anidar `if` profundos.

## Comentarios
- El mejor comentario es el que no hace falta porque el código se explica solo.
- Comenta el **porqué**, no el **qué**. Elimina comentarios obsoletos o redundantes.
- Nada de código comentado "por si acaso": eso es para el control de versiones.

## Formato y estructura
- Consistencia > preferencia personal: sigue el estilo del proyecto (linter/formatter).
- Agrupa lo relacionado; separa con líneas en blanco las ideas distintas.
- Líneas y archivos no gigantes; si un archivo hace demasiado, divídelo.

## Manejo de errores
- Usa excepciones, no códigos de retorno de error mezclados con datos.
- No tragues excepciones (`except: pass`); registra y/o re-lanza con contexto.
- Falla rápido y con mensajes claros y accionables.

## Fronteras y dependencias
- Aísla las librerías de terceros detrás de una interfaz propia cuando sea razonable.
- No dependas de detalles internos de otra capa (respeta la dirección de dependencias).

## Señales de alerta (revisar en code review)
- Funciones > ~40 líneas o con muchos niveles de indentación.
- Duplicación evidente (ver [DRY en anti_patterns_code_smells](anti_patterns_code_smells.md)).
- Nombres genéricos (`data`, `tmp`, `manager`, `util`) sin contexto.
- Números/strings mágicos repetidos.
