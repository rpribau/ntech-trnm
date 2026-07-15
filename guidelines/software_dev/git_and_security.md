---
title: Git hygiene y seguridad básica
domain: software_dev
tags: [git, seguridad, secretos, dependencias]
---

# Git hygiene y seguridad

## Git
- **Commits pequeños y atómicos**, con mensajes en imperativo que expliquen el porqué.
- Usa ramas por feature/fix; PRs revisables (no cambios gigantes de 5k líneas).
- **Nunca** commitees secretos, datos sensibles ni artefactos grandes.
- `.gitignore` para entornos, caches, datos generados, claves.
- Historia limpia: evita commits "wip"/"fix typo" sin sentido en `main`.

## Seguridad de secretos
- Credenciales y tokens **solo** en variables de entorno / gestor de secretos.
- Si un secreto se filtró en git, **rótalo** (no basta con borrarlo del último commit).
- Escaneo de secretos en CI (p. ej. gitleaks/trufflehog).

## Seguridad de código
- **Valida y sanea** toda entrada externa (usuarios, archivos, red).
- Consultas a base de datos **parametrizadas** (evita inyección SQL).
- Evita `eval`/`exec` sobre entrada no confiable; cuidado con deserialización insegura (`pickle`).
- Principio de **mínimo privilegio** (tokens de solo lectura, roles acotados).
- No expongas endpoints internos sin autenticación.

## Dependencias
- Fija versiones y revisa vulnerabilidades (`pip-audit`, Dependabot).
- Minimiza dependencias; evalúa mantenimiento y licencia antes de agregar una.

## En revisión
- Busca secretos hardcodeados, `TODO/FIXME` de seguridad, entradas sin validar,
  y dependencias desactualizadas o abandonadas.
