from __future__ import annotations

import json

from langchain_core.documents import Document

from config.settings import Settings
from ntech_agent.repomap.build import build_repo_map, load_repo_map
from ntech_agent.repomap.expansion import _ExpansionRequest, select_expansion_files
from ntech_agent.repomap.graph import graph_from_edges, rank_files
from ntech_agent.repomap.skeleton import _nest_symbols, render_repo_skeleton
from ntech_agent.repomap.tags import (
    extract_definitions,
    extract_references,
    is_notebook,
    language_for_path,
    notebook_code_to_text,
)


def test_language_for_path():
    from pathlib import Path

    assert language_for_path(Path("a/b.py")) == "python"
    assert language_for_path(Path("a/b.ipynb")) == "python"
    assert language_for_path(Path("a/b.h")) == "cpp"
    assert language_for_path(Path("a/b.cpp")) == "cpp"
    assert language_for_path(Path("a/b.jsx")) == "javascript"
    assert language_for_path(Path("a/b.rs")) is None


def test_is_notebook():
    from pathlib import Path

    assert is_notebook(Path("a/b.ipynb"))
    assert not is_notebook(Path("a/b.py"))


def _notebook_json(*, code_cells: list[str], markdown_cells: list[str] | None = None) -> str:
    cells = [{"cell_type": "code", "source": [src]} for src in code_cells]
    for src in markdown_cells or []:
        cells.append({"cell_type": "markdown", "source": [src]})
    return json.dumps({"cells": cells})


def test_notebook_code_to_text_keeps_only_code_cells():
    raw = _notebook_json(
        code_cells=["def detect_objects():\n    pass\n"],
        markdown_cells=["# Título\nEsto es prosa, no código."],
    )
    text = notebook_code_to_text(raw)
    assert "detect_objects" in text
    assert "prosa" not in text


def test_notebook_code_to_text_invalid_json_returns_empty():
    assert notebook_code_to_text("{not valid json") == ""


def test_extract_definitions_from_notebook_code():
    raw = _notebook_json(code_cells=["def get_kpi_summary():\n    pass\n"])
    text = notebook_code_to_text(raw)
    symbols = extract_definitions(text, "python")
    assert {s.name for s in symbols} == {"get_kpi_summary"}


def test_extract_definitions_python():
    src = "class Foo:\n    def bar(self):\n        pass\n\n\ndef baz():\n    pass\n"
    symbols = extract_definitions(src, "python")
    names = {s.name for s in symbols}
    assert names == {"Foo", "bar", "baz"}


def test_extract_definitions_javascript_arrow_component():
    # Patrón real de ws-br/electron-app: componentes React como const-arrow.
    src = (
        "import React from 'react';\n"
        "const Header = () => {\n"
        "  const handleClick = () => {};\n"
        "  return null;\n"
        "};\n"
        "function Footer() { return null; }\n"
        "class Sidebar extends React.Component {\n"
        "  render() { return null; }\n"
        "}\n"
    )
    symbols = extract_definitions(src, "javascript")
    names = {s.name for s in symbols}
    assert names == {"Header", "Footer", "Sidebar", "render"}
    assert "handleClick" not in names  # closure interna, no debe capturarse


def test_extract_definitions_cpp_class_and_methods():
    src = (
        "class VehicleCounter {\n"
        "public:\n"
        "    VehicleCounter(int camera_id);\n"
        "    void count(const int frame);\n"
        "};\n\n"
        "void VehicleCounter::count(const int frame) {\n"
        "    detectObjects(frame);\n"
        "}\n"
    )
    symbols = extract_definitions(src, "cpp")
    names = {s.name for s in symbols}
    assert "VehicleCounter" in names
    assert "count" in names


def test_extract_definitions_unsupported_language_returns_empty():
    assert extract_definitions("fn main() {}", "rust") == []


def test_extract_references_finds_candidates_and_skips_stopwords():
    text = "result = detect_objects(frame)\nrun()\n"
    refs = extract_references(text, {"detect_objects", "run", "unused_name"})
    assert refs == {"detect_objects"}  # "run" está en la stoplist


def test_extract_references_ignores_short_names():
    refs = extract_references("x = ab()\n", {"ab"})
    assert refs == set()  # menos de _REF_MIN_LEN caracteres


def test_rank_files_referenced_file_ranks_higher_than_isolated():
    graph = graph_from_edges(
        ["a.py", "b.py", "isolated.py"],
        [
            {"from": "a.py", "to": "b.py", "weight": 3},
            {"from": "c_missing.py", "to": "b.py", "weight": 1},  # nodo no listado, se ignora
        ],
    )
    scores = rank_files(graph, seeds=None, alpha=0.85)
    assert scores["b.py"] > scores["isolated.py"]


def test_rank_files_seed_boost_favors_seed():
    graph = graph_from_edges(["a.py", "b.py"], [])
    scores = rank_files(graph, seeds={"b.py": 10.0}, alpha=0.85)
    assert scores["b.py"] > scores["a.py"]


def test_rank_files_empty_graph_returns_empty():
    graph = graph_from_edges([], [])
    assert rank_files(graph, seeds=None, alpha=0.85) == {}


def _write(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_repo_map_excludes_vendored_libs_dir(tmp_path):
    repo_dir = tmp_path / "repos" / "demo"
    _write(repo_dir / "src" / "main.py", "def entry():\n    helper()\n")
    _write(repo_dir / "src" / "util.py", "def helper():\n    pass\n")
    _write(
        repo_dir / "cpp-motor" / "libs" / "SomeVendoredLib" / "big.cpp",
        "class Vendored {\npublic:\n    void run();\n};\n",
    )
    settings = Settings(
        _env_file=None, repos_dir=tmp_path / "repos", index_dir=tmp_path / "index"
    )

    stats = build_repo_map("demo", settings)

    assert stats["excluded_vendored"] == 1
    data = load_repo_map("demo", settings)
    paths = {f["path"] for f in data["files"]}
    assert paths == {"src/main.py", "src/util.py"}


def test_build_repo_map_excludes_gitmodules_submodule(tmp_path):
    repo_dir = tmp_path / "repos" / "demo2"
    _write(repo_dir / "app.py", "def main():\n    pass\n")
    _write(
        repo_dir / "vendored_dep" / "code.py",
        "def vendored_fn():\n    pass\n",
    )
    _write(
        repo_dir / ".gitmodules",
        '[submodule "vendored_dep"]\n\tpath = vendored_dep\n\turl = https://example.com/x.git\n',
    )
    settings = Settings(
        _env_file=None, repos_dir=tmp_path / "repos", index_dir=tmp_path / "index"
    )

    stats = build_repo_map("demo2", settings)

    assert stats["excluded_vendored"] == 1
    data = load_repo_map("demo2", settings)
    assert {f["path"] for f in data["files"]} == {"app.py"}


def test_build_repo_map_extracts_symbols_from_notebook(tmp_path):
    repo_dir = tmp_path / "repos" / "demo-nb"
    _write(
        repo_dir / "notebooks" / "analysis.ipynb",
        _notebook_json(
            code_cells=["def get_kpi_summary():\n    return helper()\n"],
            markdown_cells=["# Análisis exploratorio\nProsa que no debe parsearse."],
        ),
    )
    _write(repo_dir / "src" / "helper.py", "def helper():\n    pass\n")
    settings = Settings(
        _env_file=None, repos_dir=tmp_path / "repos", index_dir=tmp_path / "index"
    )

    stats = build_repo_map("demo-nb", settings)

    assert stats["n_files"] == 2
    assert stats["languages"] == ["python"]
    data = load_repo_map("demo-nb", settings)
    nb_file = next(f for f in data["files"] if f["path"] == "notebooks/analysis.ipynb")
    assert {s["name"] for s in nb_file["symbols"]} == {"get_kpi_summary"}
    # referencia cruzada notebook -> helper.py debe quedar reflejada en el grafo
    edge = next(e for e in data["edges"] if e["from"] == "notebooks/analysis.ipynb")
    assert edge["to"] == "src/helper.py"


def test_build_repo_map_missing_repo_returns_empty_stats(tmp_path):
    settings = Settings(
        _env_file=None, repos_dir=tmp_path / "repos", index_dir=tmp_path / "index"
    )
    stats = build_repo_map("does-not-exist", settings)
    assert stats == {
        "repo": "does-not-exist",
        "n_files": 0,
        "n_edges": 0,
        "languages": [],
        "excluded_vendored": 0,
    }


def test_build_repo_map_cross_file_reference_produces_edge(tmp_path):
    repo_dir = tmp_path / "repos" / "demo3"
    _write(repo_dir / "a.py", "def caller():\n    detect_objects()\n")
    _write(repo_dir / "b.py", "def detect_objects():\n    pass\n")
    settings = Settings(
        _env_file=None, repos_dir=tmp_path / "repos", index_dir=tmp_path / "index"
    )

    stats = build_repo_map("demo3", settings)

    assert stats["n_edges"] >= 1
    data = load_repo_map("demo3", settings)
    edge = next(e for e in data["edges"] if e["from"] == "a.py")
    assert edge["to"] == "b.py"


def _repomap_settings(tmp_path):
    return Settings(_env_file=None, repos_dir=tmp_path / "repos", index_dir=tmp_path / "index")


def test_repomap_node_replaces_retrieved_with_full_files(tmp_path, monkeypatch):
    from ntech_agent.graph.nodes import repomap_node

    repo_dir = tmp_path / "repos" / "demo"
    _write(repo_dir / "a.py", "def entry():\n    helper()\n")
    _write(repo_dir / "b.py", "def helper():\n    pass\n")
    settings = _repomap_settings(tmp_path)
    build_repo_map("demo", settings)
    monkeypatch.setattr("ntech_agent.graph.nodes.get_settings", lambda: settings)

    seed = Document(page_content="...", metadata={"repo": "demo", "path": "b.py"})
    result = repomap_node({"route": "review", "repo": "demo", "retrieved": [seed]})

    assert "retrieved" in result
    paths = {d.metadata["path"] for d in result["retrieved"]}
    assert paths == {"a.py", "b.py"}
    assert all(d.metadata["source_type"] == "repomap_file" for d in result["retrieved"])


def test_repomap_node_skips_when_route_not_relevant(tmp_path, monkeypatch):
    # summary/assist/review sí generan skeleton desde este cambio — solo rutas
    # fuera de _REPOMAP_ROUTES (ej. commits, org_report) siguen autosalteándose.
    from ntech_agent.graph.nodes import repomap_node

    settings = _repomap_settings(tmp_path)
    monkeypatch.setattr("ntech_agent.graph.nodes.get_settings", lambda: settings)

    assert repomap_node({"route": "commits", "repo": "demo", "retrieved": []}) == {}
    assert repomap_node({"route": "org_report", "repo": "demo", "retrieved": []}) == {}


def test_repomap_node_skips_when_repo_below_min_files(tmp_path, monkeypatch):
    from ntech_agent.graph.nodes import repomap_node

    repo_dir = tmp_path / "repos" / "notebooks-only"
    _write(repo_dir / "analysis.ipynb", "{}")  # sin lenguaje soportado
    settings = _repomap_settings(tmp_path)
    build_repo_map("notebooks-only", settings)
    monkeypatch.setattr("ntech_agent.graph.nodes.get_settings", lambda: settings)

    result = repomap_node({"route": "review", "repo": "notebooks-only", "retrieved": []})
    assert result == {}  # deja el retrieved chunk-based original intacto


def test_repomap_node_skips_when_disabled(tmp_path, monkeypatch):
    from ntech_agent.graph.nodes import repomap_node

    repo_dir = tmp_path / "repos" / "demo"
    _write(repo_dir / "a.py", "def entry():\n    pass\n")
    settings = Settings(
        _env_file=None,
        repos_dir=tmp_path / "repos",
        index_dir=tmp_path / "index",
        repomap_enabled=False,
    )
    build_repo_map("demo", settings)
    monkeypatch.setattr("ntech_agent.graph.nodes.get_settings", lambda: settings)

    assert repomap_node({"route": "review", "repo": "demo", "retrieved": []}) == {}


# --------------------------------------------------------------------------- #
# skeleton.py
# --------------------------------------------------------------------------- #
def test_nest_symbols_infers_class_method_containment():
    symbols = [
        {"name": "Foo", "kind": "class", "start_line": 1, "end_line": 20},
        {"name": "bar", "kind": "function", "start_line": 2, "end_line": 5},
        {"name": "baz", "kind": "function", "start_line": 25, "end_line": 30},
    ]
    nested = _nest_symbols(symbols)
    depths = {s["name"]: s["_depth"] for s in nested}
    assert depths == {"Foo": 0, "bar": 1, "baz": 0}


def test_render_repo_skeleton_lists_all_files_with_symbols(tmp_path):
    repo_dir = tmp_path / "repos" / "demo-skel"
    _write(repo_dir / "a.py", "def entry():\n    helper()\n")
    _write(repo_dir / "b.py", "def helper():\n    pass\n")
    settings = _repomap_settings(tmp_path)
    build_repo_map("demo-skel", settings)
    data = load_repo_map("demo-skel", settings)

    skeleton = render_repo_skeleton(data, settings)
    assert "a.py" in skeleton and "entry" in skeleton
    assert "b.py" in skeleton and "helper" in skeleton


def test_render_repo_skeleton_omits_files_without_symbols(tmp_path):
    repo_dir = tmp_path / "repos" / "demo-empty"
    _write(repo_dir / "has_defs.py", "def foo():\n    pass\n")
    _write(repo_dir / "no_defs.py", "x = 1\n")
    settings = _repomap_settings(tmp_path)
    build_repo_map("demo-empty", settings)
    data = load_repo_map("demo-empty", settings)

    skeleton = render_repo_skeleton(data, settings)
    assert "has_defs.py" in skeleton
    assert "no_defs.py" not in skeleton


def test_render_repo_skeleton_caps_symbols_per_file(tmp_path):
    repo_dir = tmp_path / "repos" / "demo-cap"
    _write(repo_dir / "a.py", "def one():\n    pass\n\n\ndef two():\n    pass\n")
    build_settings = _repomap_settings(tmp_path)
    build_repo_map("demo-cap", build_settings)
    data = load_repo_map("demo-cap", build_settings)

    settings = Settings(
        _env_file=None,
        repos_dir=tmp_path / "repos",
        index_dir=tmp_path / "index",
        repomap_skeleton_max_symbols_per_file=1,
    )
    skeleton = render_repo_skeleton(data, settings)
    assert "one" in skeleton
    assert "two" not in skeleton
    assert "(+1 símbolos más)" in skeleton


def test_render_repo_skeleton_truncates_files_when_over_max_files_cap(tmp_path):
    repo_dir = tmp_path / "repos" / "demo-many"
    _write(repo_dir / "a.py", "def a_fn():\n    pass\n")
    _write(repo_dir / "b.py", "def b_fn():\n    pass\n")
    _write(repo_dir / "c.py", "def c_fn():\n    pass\n")
    build_settings = _repomap_settings(tmp_path)
    build_repo_map("demo-many", build_settings)
    data = load_repo_map("demo-many", build_settings)

    settings = Settings(
        _env_file=None,
        repos_dir=tmp_path / "repos",
        index_dir=tmp_path / "index",
        repomap_skeleton_max_files=2,
    )
    skeleton = render_repo_skeleton(data, settings)
    assert skeleton != ""
    assert "a.py" in skeleton
    assert "b.py" in skeleton
    assert "c.py" not in skeleton
    assert "omitidos por límite de tamaño" in skeleton


def test_render_repo_skeleton_empty_when_no_files():
    settings = Settings(_env_file=None)
    assert render_repo_skeleton({"files": []}, settings) == ""


# --------------------------------------------------------------------------- #
# expansion.py
# --------------------------------------------------------------------------- #
class _FakeExpansionLLM:
    def __init__(self, paths):
        self._paths = paths

    def invoke(self, prompt):
        return _ExpansionRequest(paths=self._paths)


class _FakeExpansionChatModel:
    def __init__(self, paths):
        self._paths = paths

    def with_structured_output(self, model):
        return _FakeExpansionLLM(self._paths)


def test_select_expansion_files_filters_invalid_and_duplicate_paths(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.repomap.expansion.get_chat_model",
        lambda **kw: _FakeExpansionChatModel(["real.py", "real.py", "fake.py"]),
    )
    settings = Settings(_env_file=None)
    chosen = select_expansion_files(
        question="q",
        skeleton="### real.py\ndef f  [L1-2]",
        already_included=set(),
        valid_paths={"real.py"},
        settings=settings,
    )
    assert chosen == ["real.py"]


def test_select_expansion_files_respects_max_files_cap(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.repomap.expansion.get_chat_model",
        lambda **kw: _FakeExpansionChatModel(["a.py", "b.py", "c.py"]),
    )
    settings = Settings(_env_file=None, repomap_expansion_max_files=2)
    chosen = select_expansion_files(
        question="q",
        skeleton="algo",
        already_included=set(),
        valid_paths={"a.py", "b.py", "c.py"},
        settings=settings,
    )
    assert len(chosen) == 2


def test_select_expansion_files_excludes_already_included(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.repomap.expansion.get_chat_model",
        lambda **kw: _FakeExpansionChatModel(["a.py"]),
    )
    settings = Settings(_env_file=None)
    chosen = select_expansion_files(
        question="q",
        skeleton="algo",
        already_included={"a.py"},
        valid_paths={"a.py"},
        settings=settings,
    )
    assert chosen == []


def test_select_expansion_files_normalizes_paths_before_validating(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.repomap.expansion.get_chat_model",
        lambda **kw: _FakeExpansionChatModel(["./a.py"]),
    )
    settings = Settings(_env_file=None)
    chosen = select_expansion_files(
        question="q",
        skeleton="algo",
        already_included=set(),
        valid_paths={"a.py"},
        settings=settings,
    )
    assert chosen == ["a.py"]


def test_select_expansion_files_returns_empty_on_llm_failure(monkeypatch):
    def _raise(**kw):
        raise RuntimeError("backend caído")

    monkeypatch.setattr("ntech_agent.repomap.expansion.get_chat_model", _raise)
    settings = Settings(_env_file=None)
    chosen = select_expansion_files(
        question="q",
        skeleton="algo",
        already_included=set(),
        valid_paths={"a.py"},
        settings=settings,
    )
    assert chosen == []


def test_select_expansion_files_empty_skeleton_short_circuits(monkeypatch):
    def _fail_if_called(**kw):
        raise AssertionError("no debería llamar al LLM con skeleton vacío")

    monkeypatch.setattr("ntech_agent.repomap.expansion.get_chat_model", _fail_if_called)
    settings = Settings(_env_file=None)
    chosen = select_expansion_files(
        question="q", skeleton="", already_included=set(), valid_paths={"a.py"}, settings=settings
    )
    assert chosen == []


def test_select_expansion_files_disabled_setting_short_circuits(monkeypatch):
    def _fail_if_called(**kw):
        raise AssertionError("no debería llamar al LLM si expansion está deshabilitada")

    monkeypatch.setattr("ntech_agent.repomap.expansion.get_chat_model", _fail_if_called)
    settings = Settings(_env_file=None, repomap_expansion_enabled=False)
    chosen = select_expansion_files(
        question="q",
        skeleton="algo",
        already_included=set(),
        valid_paths={"a.py"},
        settings=settings,
    )
    assert chosen == []


# --------------------------------------------------------------------------- #
# repomap_node — skeleton + expansión
# --------------------------------------------------------------------------- #
def test_repomap_node_sets_skeleton_for_review_route(tmp_path, monkeypatch):
    from ntech_agent.graph.nodes import repomap_node

    repo_dir = tmp_path / "repos" / "demo"
    _write(repo_dir / "a.py", "def entry():\n    helper()\n")
    _write(repo_dir / "b.py", "def helper():\n    pass\n")
    settings = _repomap_settings(tmp_path)
    build_repo_map("demo", settings)
    monkeypatch.setattr("ntech_agent.graph.nodes.get_settings", lambda: settings)
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.select_expansion_files", lambda **kw: []
    )

    result = repomap_node({"route": "review", "repo": "demo", "retrieved": []})
    assert "entry" in result["repomap_skeleton"]
    assert "helper" in result["repomap_skeleton"]
    assert "retrieved" in result  # top-N sigue corriendo sin cambios en review


def test_repomap_node_sets_skeleton_for_summary_route_without_replacing_retrieved(
    tmp_path, monkeypatch
):
    from ntech_agent.graph.nodes import repomap_node

    repo_dir = tmp_path / "repos" / "demo"
    _write(repo_dir / "a.py", "def entry():\n    pass\n")
    settings = _repomap_settings(tmp_path)
    build_repo_map("demo", settings)
    monkeypatch.setattr("ntech_agent.graph.nodes.get_settings", lambda: settings)
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.select_expansion_files", lambda **kw: []
    )

    result = repomap_node({"route": "summary", "repo": "demo", "retrieved": []})
    assert "entry" in result["repomap_skeleton"]
    assert "retrieved" not in result  # summary no reemplaza retrieved (sin expansión)


def test_repomap_node_sets_skeleton_for_assist_route(tmp_path, monkeypatch):
    from ntech_agent.graph.nodes import repomap_node

    repo_dir = tmp_path / "repos" / "demo"
    _write(repo_dir / "a.py", "def entry():\n    pass\n")
    settings = _repomap_settings(tmp_path)
    build_repo_map("demo", settings)
    monkeypatch.setattr("ntech_agent.graph.nodes.get_settings", lambda: settings)
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.select_expansion_files", lambda **kw: []
    )

    result = repomap_node({"route": "assist", "repo": "demo", "retrieved": []})
    assert "entry" in result["repomap_skeleton"]


def test_repomap_node_skeleton_still_runs_when_below_min_files_for_rank(tmp_path, monkeypatch):
    # El test clave de la decisión de desacoplar: repo de 1 solo archivo con
    # símbolos (por debajo de repomap_min_files_for_rank=2) sigue generando
    # skeleton, aunque el top-N por PageRank no corra.
    from ntech_agent.graph.nodes import repomap_node

    repo_dir = tmp_path / "repos" / "solo"
    _write(repo_dir / "a.py", "def entry():\n    pass\n")
    settings = _repomap_settings(tmp_path)
    build_repo_map("solo", settings)
    monkeypatch.setattr("ntech_agent.graph.nodes.get_settings", lambda: settings)
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.select_expansion_files", lambda **kw: []
    )

    result = repomap_node({"route": "review", "repo": "solo", "retrieved": []})
    assert "entry" in result["repomap_skeleton"]
    assert "retrieved" not in result  # top-N no corrió (1 archivo < min_files_for_rank)


def test_repomap_node_skeleton_disabled_setting_omits_skeleton_key(tmp_path, monkeypatch):
    from ntech_agent.graph.nodes import repomap_node

    repo_dir = tmp_path / "repos" / "demo"
    _write(repo_dir / "a.py", "def entry():\n    helper()\n")
    _write(repo_dir / "b.py", "def helper():\n    pass\n")
    build_settings = _repomap_settings(tmp_path)
    build_repo_map("demo", build_settings)
    settings = Settings(
        _env_file=None,
        repos_dir=tmp_path / "repos",
        index_dir=tmp_path / "index",
        repomap_skeleton_enabled=False,
    )
    monkeypatch.setattr("ntech_agent.graph.nodes.get_settings", lambda: settings)
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.select_expansion_files", lambda **kw: []
    )

    result = repomap_node({"route": "review", "repo": "demo", "retrieved": []})
    assert "repomap_skeleton" not in result


def test_repomap_node_expansion_prepends_full_file_to_top_n_for_review(tmp_path, monkeypatch):
    from ntech_agent.graph.nodes import repomap_node

    repo_dir = tmp_path / "repos" / "demo"
    _write(repo_dir / "a.py", "def entry():\n    pass\n")
    _write(repo_dir / "rare.py", "def rarely_used():\n    pass\n")
    build_settings = _repomap_settings(tmp_path)
    build_repo_map("demo", build_settings)
    settings = Settings(
        _env_file=None,
        repos_dir=tmp_path / "repos",
        index_dir=tmp_path / "index",
        repomap_top_files=1,
    )
    monkeypatch.setattr("ntech_agent.graph.nodes.get_settings", lambda: settings)
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.select_expansion_files", lambda **kw: ["rare.py"]
    )

    result = repomap_node({"route": "review", "repo": "demo", "retrieved": []})
    paths = [d.metadata["path"] for d in result["retrieved"]]
    assert "rare.py" in paths
    assert paths[0] == "rare.py"  # antepuesto, sobrevive el corte de _format_context
    assert all(d.metadata["source_type"] == "repomap_file" for d in result["retrieved"])


def test_repomap_node_expansion_prepends_to_existing_chunks_for_summary_route(
    tmp_path, monkeypatch
):
    from langchain_core.documents import Document

    from ntech_agent.graph.nodes import repomap_node

    repo_dir = tmp_path / "repos" / "demo"
    _write(repo_dir / "a.py", "def entry():\n    pass\n")
    _write(repo_dir / "rare.py", "def rarely_used():\n    pass\n")
    settings = _repomap_settings(tmp_path)
    build_repo_map("demo", settings)
    monkeypatch.setattr("ntech_agent.graph.nodes.get_settings", lambda: settings)
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.select_expansion_files", lambda **kw: ["rare.py"]
    )

    chunk = Document(page_content="...", metadata={"repo": "demo", "path": "a.py"})
    result = repomap_node({"route": "summary", "repo": "demo", "retrieved": [chunk]})
    assert "retrieved" in result
    paths = [d.metadata["path"] for d in result["retrieved"]]
    assert paths[0] == "rare.py"  # expansión antepuesta a los chunks originales
    assert "a.py" in paths  # el chunk original de RAG se conserva


def test_repomap_node_expansion_disabled_setting_never_calls_select_expansion_files(
    tmp_path, monkeypatch
):
    from ntech_agent.graph.nodes import repomap_node

    def _fail_if_called(**kw):
        raise AssertionError("no debería llamar a select_expansion_files")

    repo_dir = tmp_path / "repos" / "demo"
    _write(repo_dir / "a.py", "def entry():\n    pass\n")
    build_settings = _repomap_settings(tmp_path)
    build_repo_map("demo", build_settings)
    settings = Settings(
        _env_file=None,
        repos_dir=tmp_path / "repos",
        index_dir=tmp_path / "index",
        repomap_expansion_enabled=False,
    )
    monkeypatch.setattr("ntech_agent.graph.nodes.get_settings", lambda: settings)
    monkeypatch.setattr("ntech_agent.graph.nodes.select_expansion_files", _fail_if_called)

    result = repomap_node({"route": "review", "repo": "demo", "retrieved": []})
    assert "repomap_skeleton" in result  # no crashea, solo se salteó la expansión
