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
`router → retrieve → repomap → static → reviewer → synthesize → END` — compiled once
and cached in `builder.py::get_graph`, persisted via a SQLite checkpointer (falls back
to `MemorySaver` if unavailable) so conversations have per-thread memory. Nodes are
pure functions of `AgentState` (a `TypedDict` in `graph/state.py`) that return partial
state updates; `repomap`, `static` and `reviewer` no-op (return `{}`) unless
`state["route"]` calls for them, rather than being conditionally wired as separate
graph edges — keep that self-skipping pattern if you add nodes.

- `supervisor.py::route_node` — classifies the query into one of
  `summary | review | org_report | assist` and fuzzy-matches ALL repos mentioned
  (`_Route.repos: list[str]`, `difflib` per name) against repos already cloned in
  `data/repos/`, deduping matches that resolve to the same repo (a typo + the exact
  name both matching `ws-arg` must not look like 2 different repos). `state["repo"]`
  (singular) is only set when exactly one repo was matched — `None` for 0 or 2+ —
  so every node written before multi-repo support (`repomap_node`, `static_node`,
  `reviewer_node`) keeps working unmodified: they naturally self-skip on a 2+-repo
  query without knowing multi-repo exists. `synthesize_node` checks `len(repos) > 1`
  first (before the `route=="review"` branch) and answers a comparison/relationship
  question across all of them instead — see "Retrieval" below for how multi-repo
  context is fetched. Falls back to keyword heuristics if structured-output
  classification fails (small/local models are unreliable at tool calling); the
  fallback never populates `repos` (pre-existing limitation, not fixed here).
- `nodes.py::repomap_node` — only runs on `route == "review"`; replaces the chunk-level
  `state["retrieved"]` with **complete files** chosen by PageRank over the repo's
  dependency graph (see "Repo map" below), boosted by whatever RAG already retrieved
  as seeds. Self-skips (leaving the chunk-based `retrieved` intact) if the repo has too
  few files with extracted symbols (e.g. a notebook-only repo) or `NTECH_REPOMAP_ENABLED`
  is off.
- `nodes.py::reviewer_node` — only runs on `route == "review"`; loads the Skill
  (`skills/code_best_practices/SKILL.md` + `rubric.yaml`), combines retrieved code +
  retrieved guidelines + static-analysis findings into one prompt, and asks for
  structured output (`_Review`/`_Finding` Pydantic models) with a free-text markdown
  fallback if structured output fails. On success it also stores the structured dump in
  `state["review_data"]` (used by the executive report — see "Reports" below); the chat
  rendering (`_render_review`) is a separate, deliberately compact view of the same data.
- Every LLM call in this codebase that uses `with_structured_output` has a
  `try/except` fallback to plain-text or pass-through behavior — this is deliberate,
  not defensive bloat, because the local/small-model backend is unreliable at
  structured output. **The fallback call itself must also be wrapped** (a second,
  inner `try/except` degrading to a static placeholder string) — a real production
  crash happened when Anthropic returned a transient `529 Overloaded` on the
  fallback call in `reviewer_node`, with nothing catching it, taking down the whole
  Streamlit app. A single layer of protection is not enough: any backend, not just
  local/small models, can fail transiently on *either* call. Preserve both layers in
  new LLM-calling code (see `reviewer_node`, `synthesize_node`,
  `report/executive.py::_synthesize_narrativa` for the pattern).

**Retrieval** (`ntech_agent/retrieval/hybrid.py`): vector search (Chroma, via
`ingest/index.py::get_vectorstore`) fused with BM25 (rebuilt from
`data/index/chunks.jsonl`, written alongside the vector index) using LangChain's
`EnsembleRetriever` (RRF, 0.5/0.5 weights), then optionally reranked by the LLM
(`rerank.py`, scored 0–1, thresholded, `NTECH_RERANK_ENABLED`/`NTECH_RERANK_THRESHOLD`).
Rerank also degrades gracefully (falls back to hybrid order) if the LLM call fails or
returns nothing above threshold — never return empty if there were candidates.
`search()` accepts `repo: str | list[str] | None` — with 2+ repos it does NOT run one
combined Chroma `$in` query, and it does NOT do one combined rerank either: it
retrieves **and reranks per repo** (retrieval: reduced-but-floored `fetch_k`,
`NTECH_RETRIEVAL_MULTI_REPO_MIN_FETCH_K`; rerank: reduced-but-floored `k`,
`NTECH_RETRIEVAL_MULTI_REPO_MIN_K`), then dedupes by id across the per-repo
results. Both a combined query AND a combined rerank were tried first and rejected
in that order — this was a real two-stage bug, not a hypothetical one: a combined
retrieval query fixes "repo B never even reaches the candidate pool," but a single
rerank call afterward can *still* drop repo B entirely, either because
`rerank.py`'s `_MAX_CANDIDATES=20` truncates the concatenated list before scoring
(order-dependent — whichever repo's candidates got appended first crowds out the
back of the list) or because the LLM's relevance threshold happens to score one
repo's chunks uniformly lower for that query. Reproduced this in production: a
combined-retrieval-only version still answered "no context on ws-br" for a
comparison query. Rerank must also run per repo with its own floor to guarantee
survival — global ranking at *either* stage reintroduces the same symptom. Cost:
retrieval and rerank calls both multiply by repo count (rerank calls now do hit the
LLM once per repo, not once total) — acceptable for the 2-5 repos a comparison
query names, revisit if the org's repo count grows much.

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

**Repo map** (`ntech_agent/repomap/`): gives the reviewer access to whole files
instead of only small RAG chunks. Computed **offline, once per repo, in
`build_index()`** — never per-query, never via an LLM call. `tags.py` walks each
supported file's AST with tree-sitter (Python, C/C++, JS/JSX, plus `.ipynb` — the
languages actually present across the org's repos today; unsupported languages
degrade gracefully, same pluggable-per-language spirit as static analysis) to extract
definitions (functions/classes/methods); cross-file **references** are approximated
by regex word-boundary matching against the set of defined names (a deliberate
simplification vs. a full AST reference system — documented as a thesis-scope
tradeoff). `graph.py` builds a file-level dependency graph from those references and
ranks files with `networkx.pagerank`; `build.py` persists the result to
`data/index/repomap/{repo}.json`. `nodes.py::repomap_node` reloads that graph per
query and recomputes a **personalized** PageRank that boosts whatever RAG already
retrieved as seeds, then reads the top-`NTECH_REPOMAP_TOP_FILES` files whole from
disk. `.ipynb` files are preprocessed by `tags.py::notebook_code_to_text` — only code
cells are kept (markdown cells dropped) and concatenated as if they were one Python
file, both for symbol/reference extraction and for what the reviewer actually sees
(showing raw notebook JSON would be useless); assumes a Python kernel, matching the
Python-first assumption already made in static analysis. Vendored/third-party code is excluded from the graph (both `.gitmodules`
submodule paths and common vendor directory names like `libs/`/`vendor/`) so it
doesn't dominate the ranking — see `ws-br/cpp-motor/libs/` for a real example.
**Gotcha**: `tree-sitter-language-pack` is pinned to `==0.13.0` deliberately — newer
1.x releases (their `process()`/`SymbolInfo` convenience API) have a real memory
corruption bug on Windows (`Node.start_point`/`end_point` return garbage, reproducibly
segfaults on real repo files); don't bump that dependency without re-verifying against
a real multi-hundred-line file first. Tree walks in `tags.py` use `tree.walk()`
(`TreeCursor`) rather than holding many `Node` objects in a Python-level stack/list —
that pattern also corrupted memory in testing, independent of the version bug above.

**GitHub integration** is two separate paths, not one: `github/client.py` +
`github/sync.py` use PyGithub + shallow git clones for the RAG corpus
(`data/repos/`); `github/mcp.py` wraps the hosted GitHub MCP server
(`langchain-mcp-adapters`) for live queries (issues/PRs/commits) that shouldn't wait
for a re-index. `sync.py` is careful never to persist the PAT in `.git/config` or
error messages (injects token into the clone URL only for the git operation, then
strips it) — preserve that when touching sync/clone logic.

**Reports** (`ntech_agent/report/`) are **executive** documents for non-technical
stakeholders, not a dump of the technical review: 5 fixed sections (Introducción,
Objetivos generales, Metodología, Principales hallazgos y conclusiones,
Recomendaciones), no file:line citations, no code jargon. `generate.py::_run_review_pipeline`
calls `retrieve_node → repomap_node → static_node → reviewer_node` **directly**, with
`route="review"`/`repo=repo` set by hand in the state dict — it deliberately bypasses
both `route_node` (no need to classify a canned prompt) and `synthesize_node` (report
generation never reads `state["answer"]`, so paying for that LLM call would be pure
waste; this also makes report generation immune to router misclassification).
`executive.py` is where the actual report is built: "Objetivos generales" and
"Metodología" are **pure Python templates** (the process is the same every run, no
need to ask an LLM and no risk of it inventing methodology detail); "Introducción"
reuses `review_data["resumen_contextualizado"]` (already generated by
`reviewer_node`, no extra call); only "Principales hallazgos y conclusiones" +
"Recomendaciones" need an LLM call, and it's **one combined call**, not one per
section, via `_synthesize_narrativa(prompt, ...)` — same
`with_structured_output` + double-layer plain-text-fallback pattern as
`reviewer_node`, but the prompt itself is passed in by the caller rather than built
inside the function: `build_repo_executive_report` uses `_NARRATIVA_PROMPT`
(summarize this one repo), `build_org_executive_report` uses a *different* constant,
`_ORG_NARRATIVA_PROMPT`, which explicitly asks the model to find patterns that repeat
**across** repos (shared stack, shared anti-patterns/gaps) instead of just
aggregating each repo's findings independently — `_org_evidence` feeds it each repo's
detected languages (`languages_by_repo`) alongside severity counts specifically so
there's raw material to spot a shared stack. This depends on `reviewer_node` having
populated `state["review_data"]` (the structured dump, not just the rendered chat
markdown) — see the LangGraph pipeline section above. Written to `data/reports/*.md`,
with best-effort PDF export via the optional `markdown-pdf` dependency (`pdf` extra).

## Guidelines & Skill corpus

`guidelines/software_dev/*.md` and `guidelines/data_science/*.md` are themselves
part of the RAG corpus (chunked and indexed with `source_type="guideline"`,
retrievable via `retrieval/hybrid.py::retrieve_guidelines`). `skills/code_best_practices/`
(`SKILL.md` + `rubric.yaml`) defines the review rubric injected verbatim into the
reviewer prompt — it is not indexed for retrieval, it's loaded directly. If you edit
the review rubric, edit these files; if you add general engineering/DS guidance
meant to be cited by the reviewer, add a guideline file under `guidelines/` and
re-run `build_index`.