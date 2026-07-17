from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document

from config.settings import Settings
from ntech_agent.graph.nodes import (
    _answer_multi_repo_question,
    _answer_review_question,
    _citations,
    _Finding,
    _format_context,
    _render_review,
    _Review,
    _review_evidence,
    synthesize_node,
)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


class _FakeLLM:
    def __init__(self, text: str = "", raise_error: bool = False, raise_structured: bool = False):
        self._text = text
        self._raise_error = raise_error
        self._raise_structured = raise_structured

    def invoke(self, prompt):
        if self._raise_error:
            raise RuntimeError("backend caído")
        return SimpleNamespace(content=self._text)

    def with_structured_output(self, model):
        if self._raise_structured:
            raise RuntimeError("structured output no soportado")
        raise NotImplementedError


def test_citations_dedupes_by_file_not_by_line():
    docs = [
        Document(page_content="a", metadata={"repo": "ws-br", "path": "a.py", "start_line": 1}),
        Document(page_content="a", metadata={"repo": "ws-br", "path": "a.py", "start_line": 42}),
        Document(page_content="a", metadata={"repo": "ws-br", "path": "b.py", "start_line": 1}),
    ]
    result = _citations(docs, _settings())
    assert result.splitlines() == ["- `ws-br/a.py`", "- `ws-br/b.py`"]


def test_citations_caps_and_notes_remaining():
    docs = [
        Document(page_content="x", metadata={"repo": "ws-br", "path": f"f{i}.py"})
        for i in range(15)
    ]
    result = _citations(docs, _settings(citations_max=12))
    lines = result.splitlines()
    assert len(lines) == 13  # 12 citas + nota de truncado
    assert lines[-1] == "_(+3 fuentes más)_"


def test_citations_empty_returns_placeholder():
    assert _citations([], _settings()) == "_(sin fuentes)_"


def test_format_context_always_keeps_repomap_file_docs_even_when_over_max_docs():
    chunks = [
        Document(page_content=f"chunk {i}", metadata={"path": f"c{i}.py"}) for i in range(6)
    ]
    repomap_docs = [
        Document(
            page_content="archivo completo uno",
            metadata={"path": "one.py", "source_type": "repomap_file"},
        ),
        Document(
            page_content="archivo completo dos",
            metadata={"path": "two.py", "source_type": "repomap_file"},
        ),
    ]
    docs = chunks + repomap_docs  # los repomap_file quedan al final, después del corte naive
    result = _format_context(docs, _settings(ctx_max_docs=3))
    assert "archivo completo uno" in result
    assert "archivo completo dos" in result


def _finding(severidad: str, ubicacion: str = "a.py:1") -> _Finding:
    return _Finding(
        ubicacion=ubicacion,
        categoria="Correctitud",
        severidad=severidad,
        hallazgo="algo",
        por_que="riesgo",
        guia="guidelines/x.md",
        recomendacion="arreglar",
    )


def test_render_review_orders_by_severity_and_caps():
    review = _Review(
        resumen_contextualizado="resumen",
        fortalezas=["bien"],
        areas_de_oportunidad=["mejorar"],
        hallazgos=[
            _finding("sugerencia"),
            _finding("critico"),
            _finding("menor"),
            _finding("mayor"),
        ],
    )
    text = _render_review(review, _settings(review_inline_findings_max=2))
    finding_lines = [line for line in text.splitlines() if line.startswith("- [")]
    assert finding_lines[0].startswith("- [critico]")
    assert finding_lines[1].startswith("- [mayor]")
    assert "_(+2 hallazgos más" in text


def test_render_review_drops_por_que_and_guia_from_chat():
    review = _Review(
        resumen_contextualizado="r",
        fortalezas=[],
        areas_de_oportunidad=[],
        hallazgos=[_finding("mayor")],
    )
    text = _render_review(review, _settings(review_inline_findings_max=6))
    assert "riesgo" not in text  # por_que ya no se muestra en el chat
    assert "guidelines/x.md" not in text  # guia ya no se muestra en el chat
    assert "arreglar" in text  # recomendacion sigue presente


def test_render_review_no_findings():
    review = _Review(
        resumen_contextualizado="r", fortalezas=[], areas_de_oportunidad=[], hallazgos=[]
    )
    text = _render_review(review, _settings())
    assert "(sin hallazgos)" in text


def test_review_evidence_includes_full_detail_for_the_llm():
    review_data = {
        "resumen_contextualizado": "resumen",
        "fortalezas": ["bien"],
        "areas_de_oportunidad": ["mejorar"],
        "hallazgos": [_finding("mayor").model_dump()],
    }
    evidence = _review_evidence(review_data, "")
    # a diferencia del render de chat, la evidencia SÍ incluye por_que/guia
    # (el LLM las necesita para responder bien; lo que se compacta es la salida).
    assert "riesgo" in evidence
    assert "guidelines/x.md" in evidence


def test_review_evidence_falls_back_to_markdown_when_no_structured_data():
    assert _review_evidence({}, "**Resumen:** algo") == "**Resumen:** algo"


def test_answer_review_question_uses_llm_response(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model",
        lambda **kwargs: _FakeLLM(text="Sí, hay funciones duplicadas en dos notebooks."),
    )
    settings = _settings()
    answer = _answer_review_question(
        "¿Hay funciones duplicadas?", {"hallazgos": []}, "", settings
    )
    assert answer == "Sí, hay funciones duplicadas en dos notebooks."


def test_answer_review_question_falls_back_to_review_markdown_on_error(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model",
        lambda **kwargs: _FakeLLM(raise_error=True),
    )
    settings = _settings()
    answer = _answer_review_question(
        "¿Hay funciones duplicadas?", {}, "**Resumen:** revisión completa", settings
    )
    assert answer == "**Resumen:** revisión completa"


def test_synthesize_node_review_route_answers_the_specific_question(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model",
        lambda **kwargs: _FakeLLM(text="Sí: `create_sequences` se repite en 2 notebooks."),
    )
    state = {
        "route": "review",
        "repo": "ws-arg",
        "question": "¿Hay funciones duplicadas?",
        "review": "**Resumen:** revisión completa con 8 hallazgos...",
        "review_data": {"hallazgos": []},
        "retrieved": [],
    }
    result = synthesize_node(state)
    assert "Sí: `create_sequences`" in result["answer"]
    assert "revisión completa con 8 hallazgos" not in result["answer"]


def test_synthesize_node_assist_route_degrades_gracefully_on_llm_failure(monkeypatch):
    # Antes de este fix, un error del backend (p. ej. un 5xx transitorio de
    # Anthropic) en esta rama no tenía ningún try/except y crasheaba el grafo.
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model",
        lambda **kwargs: _FakeLLM(raise_error=True),
    )
    state = {"route": "assist", "question": "¿qué hace esto?", "retrieved": []}
    result = synthesize_node(state)
    assert "no respondió" in result["answer"]
    assert "## Fuentes" in result["answer"]


def test_answer_multi_repo_question_uses_llm_response(monkeypatch):
    text = "ws-arg es ciencia de datos, ws-br es un sistema de conteo."
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model", lambda **kwargs: _FakeLLM(text=text)
    )
    docs = [Document(page_content="x", metadata={"repo": "ws-arg", "path": "a.ipynb"})]
    answer = _answer_multi_repo_question(
        "¿son iguales ws-arg y ws-br?", ["ws-arg", "ws-br"], docs, _settings()
    )
    assert answer == "ws-arg es ciencia de datos, ws-br es un sistema de conteo."


def test_answer_multi_repo_question_falls_back_on_error(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model",
        lambda **kwargs: _FakeLLM(raise_error=True),
    )
    answer = _answer_multi_repo_question("¿son iguales?", ["ws-arg", "ws-br"], [], _settings())
    assert "no respondió" in answer


def test_answer_multi_repo_question_logs_the_real_exception_on_failure(monkeypatch):
    # Antes de esto, la excepción real (rate limit, timeout, 5xx) quedaba
    # invisible detrás del mensaje genérico — sin ningún rastro para diagnosticar
    # por qué falló un backend que un momento antes respondía bien.
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model",
        lambda **kwargs: _FakeLLM(raise_error=True),
    )
    logged = {}
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.log_llm_failure",
        lambda where, exc: logged.update(where=where, exc=exc),
    )
    _answer_multi_repo_question("¿son iguales?", ["ws-arg", "ws-br"], [], _settings())
    assert logged["where"] == "nodes.py::_answer_multi_repo_question"
    assert isinstance(logged["exc"], RuntimeError)


def test_synthesize_node_multi_repo_takes_precedence_over_review_route(monkeypatch):
    # Antes de este fix, una pregunta multi-repo clasificada como "review" caía en
    # _answer_review_question con evidencia vacía ("no se generó revisión").
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model",
        lambda **kwargs: _FakeLLM(text="Comparación: ws-arg es Python/ML, ws-br es C++/visión."),
    )
    state = {
        "route": "review",
        "repo": None,
        "repos": ["ws-arg", "ws-br"],
        "question": "¿los proyectos de ws-arg y ws-br son iguales o distintos?",
        "review": "_(no se generó revisión)_",
        "review_data": {},
        "retrieved": [Document(page_content="x", metadata={"repo": "ws-arg", "path": "a.py"})],
    }
    result = synthesize_node(state)
    assert "Comparación: ws-arg es Python/ML" in result["answer"]
    assert "no se generó revisión" not in result["answer"]


def test_synthesize_node_single_item_repos_list_follows_normal_review_flow(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model",
        lambda **kwargs: _FakeLLM(text="Sí, hay hallazgos de seguridad."),
    )
    state = {
        "route": "review",
        "repo": "ws-arg",
        "repos": ["ws-arg"],
        "question": "¿hay hallazgos de seguridad?",
        "review": "**Resumen:** ...",
        "review_data": {"hallazgos": []},
        "retrieved": [],
    }
    result = synthesize_node(state)
    assert "Sí, hay hallazgos de seguridad." in result["answer"]


def test_synthesize_node_org_report_ignores_repos_list(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model",
        lambda **kwargs: _FakeLLM(text="Panorama de la organización: 3 repos, todos en Python."),
    )
    state = {
        "route": "org_report",
        "repo": None,
        "repos": ["ws-arg", "ws-br"],  # no debe activar la rama de comparación
        "question": "dame un panorama de la organización",
        "retrieved": [],
    }
    result = synthesize_node(state)
    assert "Panorama de la organización" in result["answer"]


def test_reviewer_node_degrades_gracefully_when_both_llm_calls_fail(monkeypatch):
    from ntech_agent.graph.nodes import reviewer_node

    monkeypatch.setattr("ntech_agent.graph.nodes.retrieve_guidelines", lambda *a, **k: [])
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model",
        lambda **kwargs: _FakeLLM(raise_error=True, raise_structured=True),
    )
    result = reviewer_node({"route": "review", "repo": "ws-arg", "retrieved": []})
    assert result["review_data"] == {}
    assert "no respondió" in result["review"]
