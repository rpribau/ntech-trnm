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
`router → retrieve → repomap → static → reviewer → commits → synthesize → END` —
compiled once and cached in `builder.py::get_graph`, persisted via a SQLite
checkpointer (falls back to `MemorySaver` if unavailable) so conversations have
per-thread memory. Nodes are pure functions of `AgentState` (a `TypedDict` in
`graph/state.py`) that return partial state updates; `repomap`, `static`, `reviewer`
and `commits` no-op (return `{}`) unless `state["route"]` calls for them, rather than
being conditionally wired as separate graph edges — keep that self-skipping pattern
if you add nodes.

- `supervisor.py::route_node` — classifies the query into one of
  `summary | review | org_report | assist | commits` and fuzzy-matches ALL repos mentioned
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
  `commits` is a distinct route from `review` — it's about activity/history ("último
  commit", "quién participa", "actividad reciente"), never code quality; the
  keyword fallback checks for it before `review`/`org_report`/`summary` so a
  transient structured-output failure on an activity question doesn't misroute it.
- `nodes.py::repomap_node` — runs on `route in {"review", "summary", "assist"}` (not just
  `review`). In all three it can populate `state["repomap_skeleton"]`, a signatures-only
  index of every file in the repo (see "Repo map" below), and can prepend a handful of
  **complete files** to `state["retrieved"]` via directed expansion. Only on `route ==
  "review"` does it additionally *replace* the chunk-level `retrieved` outright with
  **complete files** chosen by PageRank over the repo's dependency graph, boosted by
  whatever RAG already retrieved as seeds — that replacement stays review-only because
  of its cost; the skeleton/expansion is cheap enough to run on every single-repo query.
  Self-skips entirely if there's no repo, `NTECH_REPOMAP_ENABLED` is off, or the repo map
  JSON doesn't exist for the repo.
- `nodes.py::reviewer_node` — only runs on `route == "review"`; loads the Skill
  (`skills/code_best_practices/SKILL.md` + `rubric.yaml`), combines retrieved code +
  retrieved guidelines + static-analysis findings into one prompt, and asks for
  structured output (`_Review`/`_Finding` Pydantic models) with a free-text markdown
  fallback if structured output fails. On success it also stores the structured dump in
  `state["review_data"]` (used by the executive report — see "Reports" below); the chat
  rendering (`_render_review`) is a separate, deliberately compact view of the same data.
- `nodes.py::commit_insights_node` — runs on `route == "commits"`; loads deterministic
  SQL facts (last commit, contributors, weekly activity, recent commits with any
  already-generated summaries) per mentioned repo from `ntech_agent/commits/query.py`
  — no RAG, no LLM call in this node itself. `synthesize_node` special-cases
  `route == "commits"` **before** the multi-repo and `review` branches (a commits
  question about 2+ repos must answer from `commit_facts`, not fall into the
  RAG-based multi-repo comparison path) and calls `_answer_commits_question`, which
  only asks the LLM to *phrase* an answer over those already-computed facts — it
  never lets the model invent dates, authors, or commit counts. See "Commit metadata
  & insights dashboard" below for where `commit_facts` actually comes from.
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

Two more repo-map features build on that same offline-computed data and run at query
time for `route in {"review", "summary", "assist"}` (not just `review`):

- `repomap/skeleton.py::render_repo_skeleton` — a signatures-only index (no function
  bodies) of **every** file in the repo that has extracted symbols, not just the
  PageRank top-N. Reuses the `symbols` field already persisted per file in the repo map
  JSON (`build.py` writes it; nothing read it before this feature) — no extra
  tree-sitter work at query time. Solves a real blind spot of pure PageRank selection: a
  rarely-referenced file (e.g. a header defining a class interface that almost nothing
  else imports) has structurally low PageRank and never makes the top-N unless RAG
  happens to retrieve it by semantic similarity — the skeleton gives the reviewer
  visibility into that file's existence and shape regardless of ranking. Nesting (e.g. a
  method under its class) is inferred by line-range containment, since `Symbol` carries
  no parent/child relationship. Truncates gracefully — never returns an empty index just
  because a repo is large — capped by `NTECH_REPOMAP_SKELETON_MAX_FILES` (file count)
  and `NTECH_REPOMAP_SKELETON_MAX_CHARS` (total rendered size); per-file symbol count is
  capped separately by `NTECH_REPOMAP_SKELETON_MAX_SYMBOLS_PER_FILE`. Lives in its own
  `AgentState["repomap_skeleton"]` field (a plain string), deliberately *not* a
  `Document` in `retrieved` — a `Document` per file would compete for `ctx_max_docs`
  slots against real chunks/full-files and would pollute `_citations()` with one entry
  per file in the repo. Unlike the PageRank replacement above, skeleton generation is
  deliberately **not** gated by `NTECH_REPOMAP_MIN_FILES_FOR_RANK` — that threshold
  protects the *ranking's* quality (PageRank with too few nodes is noise) and doesn't
  apply to an index that doesn't rank anything; a repo with a single symbol-bearing file
  is exactly the case where showing its skeleton matters most.
- `repomap/expansion.py::select_expansion_files` — one bounded, structured-output LLM
  call (not an open-ended tool-calling loop — this codebase already treats small/local
  models as unreliable at tool-calling, see the router fallback above) that, given the
  skeleton and the user's question, can name up to `NTECH_REPOMAP_EXPANSION_MAX_FILES`
  files to read in full even though they never won the PageRank top-N nor were retrieved
  by RAG. Requested paths are normalized and validated against the repo's real file list
  before being read — never trusts the LLM's paths at face value. Unlike most
  structured-output calls in this codebase, a failure here only needs a single
  `try/except`, not the double-layer fallback described below: the result is an optional
  context enrichment the user never sees directly, so on failure it just logs and
  returns no expansion instead of needing a second LLM call to produce a fallback
  answer. Chosen files are prepended (not appended) to `retrieved` and read via the same
  `source_type: "repomap_file"` `Document` shape as the PageRank top-N, so
  `nodes.py::_format_context` and `_citations()` don't need to distinguish the two
  origins — `_format_context` treats any `"repomap_file"` doc as always-included when
  trimming to `ctx_max_docs` (filling the remaining budget with regular chunks) rather
  than relying on insertion order, since a plain prepend alone is fragile against future
  code that also prepends something before context gets formatted.

**GitHub integration** is three separate paths, not one or two — each hits a
different GitHub surface for a different reason, don't collapse them:
- `github/client.py` + `github/sync.py` — PyGithub + shallow (`--depth 1`) git clones
  for the RAG corpus (`data/repos/`); no commit history by design, since this path
  only ever feeds the current-state chunking/repo-map pipeline.
- `github/mcp.py` — wraps the hosted GitHub MCP server (`langchain-mcp-adapters`) to
  expose issues/PRs/commits/file-content as LangChain tools. **Not wired into any
  graph node today** — it's a prepared integration point, smoke-testable on its own
  via `python -m ntech_agent.github.mcp`, but no route currently calls it. Don't
  assume "live GitHub queries" happen in production; they don't yet.
- `ntech_agent/commits/` — a separate ETL against the GitHub REST API (neither the
  shallow clone nor the MCP server) that actually powers commit-history questions
  today, both in chat and in the UI. See "Commit metadata & insights dashboard" below.

`sync.py` is careful never to persist the PAT in `.git/config` or error messages
(injects token into the clone URL only for the git operation, then strips it) —
preserve that when touching sync/clone logic.

**Commit metadata & insights dashboard** (`ntech_agent/commits/`): a **fourth**,
independent SQLite database (`commits/db.py`, schema `commit_files` + `sync_state` —
distinct from both the Chroma/BM25 RAG index and the LangGraph checkpointer DB) backs
commit-activity questions in chat (`route == "commits"`) and a dedicated "Insights"
tab in the UI (`ui/insights_tab.py`). Built as its own pipeline rather than reusing
`github/sync.py`'s shallow clone (no history by design) or `github/mcp.py` (not
wired in, and not suited to bulk backfill/aggregation anyway):

- `commits/extract.py::sync_commits_for_repo` — Phase 1: pulls commits via the GitHub
  REST API (PyGithub, `since=` cursor from `sync_state.last_synced_at`, or
  `NTECH_COMMITS_LOOKBACK_DAYS` on first run) and inserts one row per
  `(repo, commit_sha, file_path)` into `commit_files` (`INSERT OR IGNORE` for
  idempotency). Uses the REST API specifically because it resolves the author's real
  GitHub login and exposes per-file `status` without needing to unshallow the local
  clone.
- `commits/summarize.py::summarize_new_commits` — Phase 2 (gated by
  `NTECH_COMMITS_SUMMARIZE_ENABLED`): one LLM call **per commit**, not per file, with
  every changed file's patch, asking for a structured `{file_path, resumen}` list —
  same double-layer `try/except` fallback pattern as `reviewer_node`, except on total
  failure it leaves `summary` NULL (retried on the next sync) instead of writing a
  placeholder, since this feeds an internal table rather than a direct user-facing
  answer. Commits touching more than `NTECH_COMMITS_MAX_FILES_PER_COMMIT` files (big
  merges, vendor dumps) are deliberately left unsummarized rather than burning LLM
  budget on them.
- `commits/query.py` — read-only SQL over `commit_files` (last commit, contributors,
  weekly activity, hotspot files), shared verbatim by `nodes.py::commit_insights_node`
  (chat) and `ui/insights_tab.py` (dashboard) so the SQL isn't duplicated in two places.
- `commits/analytics.py` — pandas transforms over the same rows (daily/weekly
  activity, top contributors, top files, change-type breakdown), used only by the
  Streamlit Insights tab; kept separate from `query.py` so aggregation logic can be
  unit-tested against synthetic DataFrames without touching SQLite.
- Sync is manual/scheduled via `python -m scripts.sync_commits` (not yet exposed as an
  `ntech-*` console script) or the "🔄 Sincronizar commits" button in the Insights tab
  — it is **not** triggered by `build_index`, and asking a "commits" question in chat
  does **not** sync on demand; `commit_insights_node` only reads whatever is already
  in `commit_files`.

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