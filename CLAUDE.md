# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A thesis project: an agent that reviews the repos of the GitHub org `NTech-TRNM`,
applies a code-quality "Skill" (design patterns, code smells, DS practices) backed
by RAG over guidelines + the org's code, and produces markdown/PDF reports with
citations. Everything is in Spanish (docstrings, prompts, CLI output, guidelines) —
match that when editing prompts, docs, or user-facing strings.

Split architecture: everything except LLM inference (embeddings, RAG, static
analysis, LangGraph orchestration, GitHub, UI) runs locally and cheaply. The **LLM
backend is swappable** via `NTECH_LLM_BACKEND` and `ntech_agent/llm.py::get_chat_model`
is the only place that knows which one is active — it returns a LangChain
`BaseChatModel`, so the rest of the code (`invoke`, `with_structured_output`) is
backend-agnostic:
- `cloudrun` — self-hosted vLLM serving `Qwen3-Coder-30B-A3B` on GCP Cloud Run
  (GPU L4, scale-to-zero); OpenAI-compatible → `ChatOpenAI` pointed at the Cloud Run
  URL with an IAM ID-token as the bearer.
- `ollama` — local model on the user's hardware; OpenAI-compatible → `ChatOpenAI`
  pointed at `localhost:11434`.
- `anthropic` — Anthropic's Claude API (`langchain_anthropic.ChatAnthropic`); **not**
  OpenAI-compatible, but same LangChain interface. Note: Opus 4.6+/Sonnet 5/Fable 5
  reject `temperature` (400), so the Anthropic branch omits it deliberately — do not
  re-add sampling params there.

## Setup & commands

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env   # fill in values (never commit .env)
```

Pipeline (run in order for a fresh environment):

```powershell
python -m scripts.sync_repos      # clone/pull all org repos into data/repos/
python -m scripts.build_index     # chunk + embed code and guidelines into Chroma
streamlit run ui/streamlit_app.py # chat UI
python -m scripts.run_review --repo <name>   # single-repo markdown report
python -m scripts.run_review --org           # org-wide report
python -m scripts.run_review --ask "..."     # one-off query, no report file
python -m scripts.eval             # recall@k / MRR over tests/gold.jsonl (retrieval only)
```

Also installed as console scripts: `ntech-sync`, `ntech-index`, `ntech-review`, `ntech-eval`.

Tests / lint:
```powershell
pytest                       # testpaths = tests/
pytest tests/test_chunking.py::test_chunk_ids_are_deterministic  # single test
ruff check .                 # select = E,F,I,UP,B; line-length 100; py311
```

Switching LLM backend: set `NTECH_LLM_BACKEND` in `.env` to `cloudrun`, `ollama`,
or `anthropic` — no code changes needed. `ollama` needs a local model pulled
(`ollama pull qwen2.5-coder:7b`); `anthropic` needs `NTECH_ANTHROPIC_API_KEY` (a
console.anthropic.com key with prepaid credit — separate from a Claude Pro/Max
subscription). Smoke-test the active backend with `python -m ntech_agent.llm`.

Deploying/updating the cloud model (`deploy/deploy.sh`, GCP-billed, semi-irreversible
— confirm with the user before running):
```bash
cd deploy && bash deploy.sh
```
See `deploy/README.md` for GPU quota, IAM (`--no-allow-unauthenticated` + ID-token auth
via `ntech-invoker` SA), quantization, and cold-start notes.

## Architecture

**Config is the single source of truth.** `config/settings.py` defines a Pydantic
`Settings` (env prefix `NTECH_`, loaded from `.env`). Nothing should be hardcoded
outside it; new tunables belong there, mirrored in `.env.example`. `get_settings()`
is `lru_cache`d — pass `settings=` explicitly through function signatures rather than
re-reading env vars, and construct a fresh `Settings()` in tests instead of mutating
the cached singleton.

**LangGraph pipeline** (`ntech_agent/graph/`): a linear graph —
`router → retrieve → static → reviewer → synthesize → END` — compiled once and
cached in `builder.py::get_graph`, persisted via a SQLite checkpointer (falls back
to `MemorySaver` if unavailable) so conversations have per-thread memory. Nodes are
pure functions of `AgentState` (a `TypedDict` in `graph/state.py`) that return partial
state updates; `static` and `reviewer` no-op (return `{}`) unless `state["route"]`
calls for them, rather than being conditionally wired as separate graph edges — keep
that self-skipping pattern if you add nodes.

- `supervisor.py::route_node` — classifies the query into one of
  `summary | review | org_report | assist` and fuzzy-matches a target repo name
  (`difflib`) against repos already cloned in `data/repos/`. Falls back to keyword
  heuristics if structured-output classification fails (small/local models are
  unreliable at tool calling).
- `nodes.py::reviewer_node` — only runs on `route == "review"`; loads the Skill
  (`skills/code_best_practices/SKILL.md` + `rubric.yaml`), combines retrieved code +
  retrieved guidelines + static-analysis findings into one prompt, and asks for
  structured output (`_Review`/`_Finding` Pydantic models) with a free-text markdown
  fallback if structured output fails.
- Every LLM call in this codebase that uses `with_structured_output` has a
  `try/except` fallback to plain-text or pass-through behavior — this is deliberate,
  not defensive bloat, because the local/small-model backend is unreliable at
  structured output. Preserve this pattern in new LLM-calling code.

**Retrieval** (`ntech_agent/retrieval/hybrid.py`): vector search (Chroma, via
`ingest/index.py::get_vectorstore`) fused with BM25 (rebuilt from
`data/index/chunks.jsonl`, written alongside the vector index) using LangChain's
`EnsembleRetriever` (RRF, 0.5/0.5 weights), then optionally reranked by the LLM
(`rerank.py`, scored 0–1, thresholded, `NTECH_RERANK_ENABLED`/`NTECH_RERANK_THRESHOLD`).
Rerank also degrades gracefully (falls back to hybrid order) if the LLM call fails or
returns nothing above threshold — never return empty if there were candidates.

**Ingestion** (`ntech_agent/ingest/chunking.py`): walks `data/repos/*` and
`guidelines/**/*.md`, chunks by language using LangChain's language-aware splitters
(mapped per extension in `_CODE_LANG`), and produces `Document`s with deterministic
UUIDv5 ids (`uuid5(namespace, f"{repo}::{path}::{chunk_index}")`) so re-indexing is
idempotent — never change that ID scheme without also handling migration of the
existing Chroma collection. `build_index.py` recreates the collection by default
(`reset=True`); use `--no-reset` to append.

**Static analysis** (`ntech_agent/analysis/static.py`): pluggable per-language
runners in a `_RUNNERS` dict (currently only `python`: ruff + radon cc/mi). Adding a
language means adding a runner function + registering it in `_RUNNERS`, plus a
matching detection branch in `run_static_analysis`. Output feeds the reviewer node
as "objective evidence" alongside RAG context and guidelines.

**GitHub integration** is two separate paths, not one: `github/client.py` +
`github/sync.py` use PyGithub + shallow git clones for the RAG corpus
(`data/repos/`); `github/mcp.py` wraps the hosted GitHub MCP server
(`langchain-mcp-adapters`) for live queries (issues/PRs/commits) that shouldn't wait
for a re-index. `sync.py` is careful never to persist the PAT in `.git/config` or
error messages (injects token into the clone URL only for the git operation, then
strips it) — preserve that when touching sync/clone logic.

**Reports** (`ntech_agent/report/generate.py`) are just `run_agent(...)` calls with
canned prompts (`route=review` per repo, or a panorama prompt for the whole org),
written to `data/reports/*.md`, with best-effort PDF export via the optional
`markdown-pdf` dependency (`pdf` extra).

## Guidelines & Skill corpus

`guidelines/software_dev/*.md` and `guidelines/data_science/*.md` are themselves
part of the RAG corpus (chunked and indexed with `source_type="guideline"`,
retrievable via `retrieval/hybrid.py::retrieve_guidelines`). `skills/code_best_practices/`
(`SKILL.md` + `rubric.yaml`) defines the review rubric injected verbatim into the
reviewer prompt — it is not indexed for retrieval, it's loaded directly. If you edit
the review rubric, edit these files; if you add general engineering/DS guidance
meant to be cited by the reviewer, add a guideline file under `guidelines/` and
re-run `build_index`.