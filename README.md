# NTech Code Review Agent

Agente que revisa los repositorios de la org de GitHub **NTech-TRNM**, aplica una
**Skill de buenas prácticas de código** (design patterns, code smells) apoyada en
**guidelines de Software Development y Ciencia de Datos**, y produce **reportes**
con áreas de oportunidad y estado contextualizado de cada repo.

Basado en la arquitectura de [RAG_Queries_Agent](https://github.com/Jorge-Polanco-Roque/RAG_Queries_Agent)
(LangGraph supervisor + RAG híbrido + clientes compatibles OpenAI).

## Arquitectura (split cloud / local + backend de LLM intercambiable)

Todo el driver (LangGraph, RAG, GitHub, UI) corre siempre local. El **backend
del LLM** es intercambiable con una sola variable de entorno
(`NTECH_LLM_BACKEND`) — el resto del código no sabe ni le importa cuál está
activo:

```
┌─────────────────────────── Local (tu Windows) ───────────────────────────┐
│  Streamlit UI  ──►  LangGraph supervisor                                  │
│                       ├─ retriever  ──►  Chroma (RAG híbrido + rerank)    │
│                       ├─ repo map   ──►  archivos completos por PageRank  │
│                       │                   (tree-sitter, offline)          │
│                       ├─ reviewer   ──►  Skill + guidelines + análisis    │
│                       │                   estático (ruff/radon)           │
│                       └─ synthesizer ─►  respuesta compacta con citas     │
│  Embeddings locales (bge-m3)   GitHub sync (clonar/pull) + GitHub MCP     │
│  Reportes ejecutivos (5 secciones, sin jerga) ──► data/reports/*.md       │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 │  NTECH_LLM_BACKEND = cloudrun | ollama | anthropic
              ┌──────────────────────────────┼──────────────────────────────┐
              ▼                              ▼                              ▼
┌─────────────────────────┐   ┌───────────────────────┐   ┌───────────────────────────┐
│ GCP Cloud Run (GPU L4)  │   │ Ollama (local)        │   │ API de Anthropic (Claude) │
│ vLLM + Qwen3-Coder-30B  │   │ modelo pequeño en tu   │   │ pay-per-token, sin infra  │
│ self-hosted, scale-to-0 │   │ hardware, offline      │   │ que mantener              │
└─────────────────────────┘   └───────────────────────┘   └───────────────────────────┘
```

| Backend | Cuándo usarlo | Costo |
|---|---|---|
| `cloudrun` | Modelo open-weight self-hosted (el ángulo de tesis: LLM propio, no comercial) | GPU L4 ~$0.67/hr solo mientras infiere; scale-to-zero en reposo |
| `ollama` | Desarrollo/demo local, sin gastar nada, sin depender de la nube | Gratis (tu hardware) |
| `anthropic` | Máxima calidad/confiabilidad sin mantener infraestructura | Pay-per-token vía [console.anthropic.com](https://console.anthropic.com) (cuenta separada de Claude Pro/Max) |

## Requisitos

- Python 3.11+
- Cuenta de GCP con facturación (usa créditos si tienes) y `gcloud` CLI
- Un **PAT de GitHub** con lectura de los repos de la org
- (Opcional, para el fallback local) [Ollama](https://ollama.com)

## Instalación (driver local)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env   # y rellena los valores
```

## Puesta en marcha

### 1. Desplegar el modelo en GCP (una vez)

```bash
cd deploy
bash deploy.sh   # sube los pesos a GCS y despliega el Cloud Run con GPU L4
```

Copia la URL resultante a `NTECH_CLOUDRUN_URL` en tu `.env`.
Ver [deploy/README.md](deploy/README.md) para el detalle y la seguridad (IAM).

### 2. Sincronizar e indexar los repos

```powershell
python -m scripts.sync_repos      # clona/actualiza todos los repos de la org
python -m scripts.build_index     # indexa código + guidelines en Chroma, y calcula el repo map
```

### 3. Usar el agente

```powershell
# UI de chat
streamlit run ui/streamlit_app.py

# o generar un reporte por CLI
python -m scripts.run_review --repo ws-arg
python -m scripts.run_review --org           # reporte de toda la organización
```

Ejemplo de Q&A en la UI: *"Dame un resumen de ws-arg"* → resumen contextual del
repo con citas a los archivos relevantes.

## Cambiar de backend de LLM

Solo edita `NTECH_LLM_BACKEND` en `.env` — el resto del código no cambia.

**Ollama (local, gratis, offline):**
```powershell
ollama pull qwen2.5-coder:7b
```
```
NTECH_LLM_BACKEND=ollama
```

**Anthropic (API de Claude, pay-per-token):**
1. Crea una cuenta en [console.anthropic.com](https://console.anthropic.com) (es
   **independiente** de una suscripción Claude Pro/Max — no la incluye) y agrega
   crédito (mínimo $5).
2. Genera una API key y ponla en `.env`:
```
NTECH_LLM_BACKEND=anthropic
NTECH_ANTHROPIC_API_KEY=sk-ant-...
```

**Cloud Run (self-hosted, ver sección "Puesta en marcha" arriba):**
```
NTECH_LLM_BACKEND=cloudrun
```

Prueba rápida del backend activo:
```powershell
python -m ntech_agent.llm
```

## Costo (tesis)

Con **scale-to-zero**, en reposo pagas ≈ solo el storage GCS de los pesos
(centavos). La GPU L4 (~$0.67/hr) se cobra **solo mientras el modelo infiere**.
Para corridas batch largas, considera `min-instances=1` temporal y apágalo al terminar.

## Estructura

```
ntech-trnm/
├── config/settings.py       # configuración tipada (Pydantic Settings)
├── ntech_agent/             # driver: LLM, RAG, repo map, GitHub, grafo, reporte
│   └── repomap/             # grafo de dependencias (tree-sitter) + PageRank, offline
├── guidelines/              # guidelines de Software Dev y Ciencia de Datos (RAG + MCP)
├── skills/code_best_practices/  # la "Skill" de revisión (rúbrica + design patterns)
├── deploy/                  # imagen vLLM + Cloud Run (GPU L4)
├── ui/streamlit_app.py      # frontend v1
├── scripts/                 # sync, build_index, run_review, eval
└── tests/
```

## Evaluación (tesis)

```powershell
python -m scripts.eval        # recall@k y MRR sobre el set gold (tests/gold.jsonl)
```
