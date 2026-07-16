from __future__ import annotations

from config.settings import Settings
from ntech_agent.graph.supervisor import _Route, route_node


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


class _FakeStructuredLLM:
    def __init__(self, result: _Route):
        self._result = result

    def invoke(self, prompt):
        return self._result


class _FakeLLM:
    def __init__(self, result: _Route | None = None, raise_error: bool = False):
        self._result = result
        self._raise_error = raise_error

    def with_structured_output(self, model):
        if self._raise_error:
            raise RuntimeError("structured output no soportado")
        return _FakeStructuredLLM(self._result)


def _route_with(monkeypatch, question: str, known: list[str], result: _Route | None, **kwargs):
    monkeypatch.setattr("ntech_agent.graph.supervisor.get_settings", lambda: _settings(**kwargs))
    monkeypatch.setattr("ntech_agent.graph.supervisor.list_local_repos", lambda settings: known)
    monkeypatch.setattr(
        "ntech_agent.graph.supervisor.get_chat_model", lambda **kw: _FakeLLM(result=result)
    )
    return route_node({"question": question})


def test_route_node_single_repo_sets_repo_and_repos(monkeypatch):
    result = _route_with(
        monkeypatch,
        "resumen de ws-arg",
        ["ws-arg", "ws-br", "ws-mex"],
        _Route(route="summary", repos=["ws-arg"]),
    )
    assert result["repo"] == "ws-arg"
    assert result["repos"] == ["ws-arg"]


def test_route_node_multi_repo_sets_repos_and_leaves_repo_none(monkeypatch):
    result = _route_with(
        monkeypatch,
        "diferencia entre ws-arg y ws-br",
        ["ws-arg", "ws-br", "ws-mex"],
        _Route(route="assist", repos=["ws-arg", "ws-br"]),
    )
    assert result["repo"] is None
    assert result["repos"] == ["ws-arg", "ws-br"]


def test_route_node_dedupes_repos_that_match_the_same_known_repo(monkeypatch):
    # "wsarg" (typo/fuzzy) y "ws-arg" (exacto) matchean al mismo repo conocido —
    # no debe degradar a "sin repo" una pregunta que en realidad es de un solo repo.
    result = _route_with(
        monkeypatch,
        "revisa ws-arg (wsarg)",
        ["ws-arg", "ws-br"],
        _Route(route="review", repos=["ws-arg", "wsarg"]),
    )
    assert result["repos"] == ["ws-arg"]
    assert result["repo"] == "ws-arg"


def test_route_node_fallback_heuristic_detects_mentioned_repo_names(monkeypatch):
    # Antes de este fix, el fallback heurístico NUNCA poblaba "repos" (limitación
    # documentada) — cualquier falla transitoria del LLM en una pregunta multi-repo
    # perdía en silencio las garantías de piso por repo en retrieval, degradando a
    # una búsqueda global sin fairness entre repos.
    monkeypatch.setattr("ntech_agent.graph.supervisor.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "ntech_agent.graph.supervisor.list_local_repos", lambda settings: ["ws-arg", "ws-br"]
    )
    monkeypatch.setattr(
        "ntech_agent.graph.supervisor.get_chat_model",
        lambda **kw: _FakeLLM(raise_error=True),
    )

    result = route_node({"question": "revisa ws-arg y ws-br por favor"})

    assert result["route"] == "review"  # matchea la keyword "revisa" en el fallback
    assert result["repos"] == ["ws-arg", "ws-br"]
    assert result["repo"] is None  # 2 repos -> repo singular queda None


def test_route_node_fallback_heuristic_single_repo_summary_unaffected(monkeypatch):
    monkeypatch.setattr("ntech_agent.graph.supervisor.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "ntech_agent.graph.supervisor.list_local_repos", lambda settings: ["ws-arg", "ws-br"]
    )
    monkeypatch.setattr(
        "ntech_agent.graph.supervisor.get_chat_model",
        lambda **kw: _FakeLLM(raise_error=True),
    )

    result = route_node({"question": "dame un resumen de ws-arg"})

    assert result["route"] == "summary"
    assert result["repos"] == ["ws-arg"]
    assert result["repo"] == "ws-arg"


def test_route_node_fallback_heuristic_org_report_takes_priority_over_resumen(monkeypatch):
    # Bug real: "resumen de todos los repositorios" contiene la palabra "resumen",
    # así que si el chequeo de "summary" corre ANTES que el de "org_report" en el
    # elif, esta consulta caía en "summary" (con repos=[]) en vez de "org_report".
    monkeypatch.setattr("ntech_agent.graph.supervisor.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "ntech_agent.graph.supervisor.list_local_repos", lambda settings: ["ws-arg", "ws-br"]
    )
    monkeypatch.setattr(
        "ntech_agent.graph.supervisor.get_chat_model",
        lambda **kw: _FakeLLM(raise_error=True),
    )

    result = route_node(
        {"question": "podrias darme un resumen de todos los repositorios que tenemos disponibles"}
    )

    assert result["route"] == "org_report"


def test_route_node_commits_route_from_structured_output(monkeypatch):
    result = _route_with(
        monkeypatch,
        "¿cuándo fueron los últimos cambios a ws-arg?",
        ["ws-arg", "ws-br"],
        _Route(route="commits", repos=["ws-arg"]),
    )
    assert result["route"] == "commits"
    assert result["repo"] == "ws-arg"


def test_route_node_commits_fallback_heuristic_matches_ultimos_cambios(monkeypatch):
    monkeypatch.setattr("ntech_agent.graph.supervisor.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "ntech_agent.graph.supervisor.list_local_repos", lambda settings: ["ws-arg", "ws-br"]
    )
    monkeypatch.setattr(
        "ntech_agent.graph.supervisor.get_chat_model",
        lambda **kw: _FakeLLM(raise_error=True),
    )

    result = route_node({"question": "¿cuándo fueron los últimos cambios a ws-arg?"})
    assert result["route"] == "commits"


def test_route_node_commits_fallback_heuristic_matches_quien_participa(monkeypatch):
    monkeypatch.setattr("ntech_agent.graph.supervisor.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "ntech_agent.graph.supervisor.list_local_repos", lambda settings: ["ws-arg", "ws-br"]
    )
    monkeypatch.setattr(
        "ntech_agent.graph.supervisor.get_chat_model",
        lambda **kw: _FakeLLM(raise_error=True),
    )

    result = route_node({"question": "¿quiénes participan en ws-arg?"})
    assert result["route"] == "commits"


def test_route_node_unmatched_repo_name_kept_raw(monkeypatch):
    # _match_repo devuelve el nombre crudo si no matchea nada conocido (comportamiento
    # preexistente) — confirmamos que la lista lo conserva igual.
    result = _route_with(
        monkeypatch,
        "compara ws-arg con un repo inventado",
        ["ws-arg", "ws-br"],
        _Route(route="assist", repos=["ws-arg", "repo-inventado"]),
    )
    assert result["repos"] == ["ws-arg", "repo-inventado"]
    assert result["repo"] is None
