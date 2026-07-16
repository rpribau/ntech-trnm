from __future__ import annotations

from types import SimpleNamespace

from config.settings import Settings
from ntech_agent.graph.nodes import (
    _answer_commits_question,
    _commit_facts_evidence,
    _recent_commits_block,
    commit_insights_node,
    synthesize_node,
)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


class _FakeLLM:
    def __init__(self, text: str = "", raise_error: bool = False):
        self._text = text
        self._raise_error = raise_error

    def invoke(self, prompt):
        if self._raise_error:
            raise RuntimeError("backend caído")
        return SimpleNamespace(content=self._text)


def test_commit_insights_node_self_skips_when_route_is_not_commits():
    result = commit_insights_node({"route": "review", "repo": "ws-arg"})
    assert result == {}


def test_commit_insights_node_returns_empty_facts_when_no_repo_matched():
    result = commit_insights_node({"route": "commits", "repos": []})
    assert result == {"commit_facts": {}}


def test_commit_insights_node_calls_org_insights_for_matched_repos(monkeypatch):
    captured = {}

    def fake_org_insights(repos, settings):
        captured["repos"] = repos
        return {r: {"total_commits": 1} for r in repos}

    monkeypatch.setattr("ntech_agent.graph.nodes.org_insights", fake_org_insights)
    result = commit_insights_node({"route": "commits", "repos": ["ws-arg", "ws-br"]})
    assert captured["repos"] == ["ws-arg", "ws-br"]
    assert result == {
        "commit_facts": {"ws-arg": {"total_commits": 1}, "ws-br": {"total_commits": 1}}
    }


def test_commit_facts_evidence_empty():
    text = _commit_facts_evidence({})
    assert "sin datos de commits" in text.lower()


def test_recent_commits_block_groups_multi_file_commit_into_one_line():
    # Antes de este fix, cada archivo era una fila suelta sin agrupar por commit
    # — el LLM no podía dar "detalle del último commit" (archivos + resumen)
    # de un vistazo.
    recent = [
        {
            "commit_sha": "abc12345678",
            "file_path": "a.py",
            "author": "dev-a",
            "committed_at": "2026-06-01T00:00:00+00:00",
            "commit_message": "fix",
            "summary": "Corrige A.",
        },
        {
            "commit_sha": "abc12345678",
            "file_path": "b.py",
            "author": "dev-a",
            "committed_at": "2026-06-01T00:00:00+00:00",
            "commit_message": "fix",
            "summary": "Corrige B.",
        },
    ]
    block = _recent_commits_block(recent)
    lines = block.splitlines()
    assert len(lines) == 1  # un commit -> una línea, no una por archivo
    assert "abc12345" in lines[0]  # sha corto
    assert "a.py (Corrige A.)" in lines[0]
    assert "b.py (Corrige B.)" in lines[0]


def test_recent_commits_block_empty():
    assert "sin commits recientes" in _recent_commits_block([])


def test_commit_facts_evidence_includes_per_commit_detail_for_author_and_last_commit_questions():
    # Reproduce el caso reportado: "¿qué commits hizo AlejandroSparza21?" y "más
    # detalles del último commit" eran irrespondibles porque la evidencia solo
    # tenía conteos agregados, nunca el listado commit por commit.
    facts = {
        "ws-mex": {
            "total_commits": 2,
            "last_commit": {
                "committed_at": "2026-06-04T00:00:00+00:00",
                "author": "rpribau",
                "commit_message": "feat: mlflow",
            },
            "contributors": [
                {"author": "rpribau", "commits": 1},
                {"author": "AlejandroSparza21", "commits": 1},
            ],
            "recent": [
                {
                    "commit_sha": "sha2later",
                    "file_path": "y.py",
                    "author": "rpribau",
                    "committed_at": "2026-06-04T00:00:00+00:00",
                    "commit_message": "feat: mlflow",
                    "summary": None,
                },
                {
                    "commit_sha": "sha1older",
                    "file_path": "x.py",
                    "author": "AlejandroSparza21",
                    "committed_at": "2026-06-03T00:00:00+00:00",
                    "commit_message": "wip",
                    "summary": "Ajuste de parámetros.",
                },
            ],
        }
    }
    evidence = _commit_facts_evidence(facts)
    assert "AlejandroSparza21" in evidence
    assert "Ajuste de parámetros." in evidence
    assert "x.py" in evidence
    assert "y.py" in evidence


def test_commit_facts_evidence_formats_last_commit_and_contributors():
    facts = {
        "ws-arg": {
            "total_commits": 3,
            "last_commit": {
                "committed_at": "2026-07-10T00:00:00+00:00",
                "author": "rpribau",
                "commit_message": "fix bug",
            },
            "contributors": [{"author": "rpribau", "commits": 3}],
        }
    }
    text = _commit_facts_evidence(facts)
    assert "ws-arg" in text
    assert "rpribau" in text
    assert "fix bug" in text


def test_answer_commits_question_uses_llm_response(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model",
        lambda **kwargs: _FakeLLM(text="El último commit de ws-arg fue el 2026-07-10."),
    )
    answer = _answer_commits_question(
        "¿cuándo fue el último cambio a ws-arg?", {"ws-arg": {"total_commits": 1}}, _settings()
    )
    assert "2026-07-10" in answer


def test_answer_commits_question_degrades_gracefully_on_llm_failure(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model",
        lambda **kwargs: _FakeLLM(raise_error=True),
    )
    answer = _answer_commits_question("¿quién participa?", {}, _settings())
    assert "no respondió" in answer


def test_synthesize_node_commits_route_skips_rag_citations(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model",
        lambda **kwargs: _FakeLLM(text="rpribau es el único contribuidor de ws-arg."),
    )
    state = {
        "route": "commits",
        "question": "¿quién participa en ws-arg?",
        "commit_facts": {"ws-arg": {"total_commits": 1}},
        "retrieved": [],
    }
    result = synthesize_node(state)
    assert "rpribau es el único contribuidor" in result["answer"]
    assert "## Fuentes" not in result["answer"]


def test_synthesize_node_commits_route_takes_precedence_over_multi_repo_branch(monkeypatch):
    # "commits" con 2+ repos no debe caer en _answer_multi_repo_question (que usa
    # RAG, no commit_facts).
    monkeypatch.setattr(
        "ntech_agent.graph.nodes.get_chat_model",
        lambda **kwargs: _FakeLLM(text="Actividad de ws-arg y ws-br."),
    )
    state = {
        "route": "commits",
        "repo": None,
        "repos": ["ws-arg", "ws-br"],
        "question": "¿quién participa en ws-arg y ws-br?",
        "commit_facts": {"ws-arg": {}, "ws-br": {}},
        "retrieved": [],
    }
    result = synthesize_node(state)
    assert "Actividad de ws-arg y ws-br." in result["answer"]
