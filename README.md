# NTech Code Review Agent

Agente que revisa los repositorios de la org de GitHub **NTech-TRNM**, aplica una
**Skill de buenas prácticas de código** (design patterns, code smells) apoyada en
**guidelines de Software Development y Ciencia de Datos**, y produce **reportes**
con áreas de oportunidad y estado contextualizado de cada repo.

Basado en la arquitectura de [RAG_Queries_Agent](https://github.com/Jorge-Polanco-Roque/RAG_Queries_Agent)
(LangGraph supervisor + RAG híbrido + clientes compatibles OpenAI).

## Arquitectura (split cloud / local)

Solo la inferencia pesada del LLM vive en GCP; el resto corre local y barato.

```
┌─────────────────────────── Local (tu Windows) ───────────────────────────┐
│  Streamlit UI  ──►  LangGraph supervisor                                  │
│                       ├─ retriever  ──►  Chroma (RAG híbrido + rerank)    │
│                       ├─ reviewer   ──►  Skill + guidelines + análisis    │
│                       │                   estático (ruff/radon)           │
│                       └─ synthesizer ─►  reporte con citas                │
│  Embeddings locales (bge-m3)   GitHub sync (clonar/pull) + GitHub MCP     │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 │ HTTPS + ID token (IAM)
                                 ▼
┌─────────────────────── GCP Cloud Run (GPU L4, scale-to-zero) ─────────────┐
│  vLLM  ──►  Qwen3-Coder-30B-A3B  (API compatible OpenAI)                  │
│  Pesos en bucket GCS (Cloud Storage FUSE).  Paga solo mientras infiere.   │
└───────────────────────────────────────────────────────────────────────────┘
```

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
python -m scripts.build_index     # indexa código + guidelines en Chroma
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

## Desarrollo barato / offline (sin GCP)

Pon `NTECH_LLM_BACKEND=ollama` en `.env` y corre un modelo chico:

```powershell
ollama pull qwen2.5-coder:7b
```

El código no cambia (ambos backends son compatibles con la API de OpenAI).

## Costo (tesis)

Con **scale-to-zero**, en reposo pagas ≈ solo el storage GCS de los pesos
(centavos). La GPU L4 (~$0.67/hr) se cobra **solo mientras el modelo infiere**.
Para corridas batch largas, considera `min-instances=1` temporal y apágalo al terminar.

## Estructura

```
ntech-trnm/
├── config/settings.py       # configuración tipada (Pydantic Settings)
├── ntech_agent/             # driver: LLM, RAG, GitHub, grafo, reporte
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
