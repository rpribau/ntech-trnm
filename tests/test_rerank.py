from __future__ import annotations

from langchain_core.documents import Document
from pydantic import ValidationError

from config.settings import Settings
from ntech_agent.retrieval.rerank import _repair_stringified_scores, _RerankResult, rerank


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _validation_error_for(scores_payload) -> ValidationError:
    try:
        _RerankResult(scores=scores_payload)
    except ValidationError as exc:
        return exc
    raise AssertionError("se esperaba que scores_payload disparara un ValidationError")


def test_repair_stringified_scores_recovers_object_wrapped_in_string():
    # Caso real observado en producción: el modelo devolvió el objeto
    # {"scores": [...]} ENTERO como string, en vez de la lista directamente.
    payload = '{"scores":[{"index":0,"score":0.9},{"index":11,"score":0.05}]}'
    exc = _validation_error_for(payload)
    repaired = _repair_stringified_scores(exc)
    assert repaired is not None
    assert [s.index for s in repaired.scores] == [0, 11]
    assert [s.score for s in repaired.scores] == [0.9, 0.05]


def test_repair_stringified_scores_recovers_bare_list_string():
    payload = '[{"index":0,"score":0.9}]'
    exc = _validation_error_for(payload)
    repaired = _repair_stringified_scores(exc)
    assert repaired is not None
    assert repaired.scores[0].index == 0


def test_repair_stringified_scores_returns_none_for_unparseable_string():
    exc = _validation_error_for("esto no es json")
    assert _repair_stringified_scores(exc) is None


class _FakeStructuredLLMBadPayload:
    def __init__(self, payload):
        self._payload = payload

    def invoke(self, prompt):
        return _RerankResult(scores=self._payload)  # dispara ValidationError


class _FakeLLM:
    def __init__(self, payload):
        self._payload = payload

    def with_structured_output(self, model):
        return _FakeStructuredLLMBadPayload(self._payload)


def test_rerank_recovers_from_stringified_scores_end_to_end(monkeypatch):
    payload = '{"scores":[{"index":0,"score":0.9},{"index":1,"score":0.05}]}'
    monkeypatch.setattr(
        "ntech_agent.retrieval.rerank.get_chat_model", lambda **kw: _FakeLLM(payload)
    )
    docs = [Document(page_content="relevante"), Document(page_content="irrelevante")]
    result = rerank("consulta", docs, k=5, settings=_settings(rerank_threshold=0.3))
    assert result == [docs[0]]  # solo el índice 0 (score 0.9) supera el umbral 0.3


def test_rerank_falls_back_when_repair_also_fails(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.retrieval.rerank.get_chat_model",
        lambda **kw: _FakeLLM("esto no es json"),
    )
    docs = [Document(page_content="a"), Document(page_content="b")]
    result = rerank("consulta", docs, k=5, settings=_settings())
    assert result == docs  # fallback: preserva el orden original
