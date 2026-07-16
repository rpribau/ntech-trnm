from __future__ import annotations

from langchain_core.documents import Document

from config.settings import Settings
from ntech_agent.retrieval import hybrid


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_dedupe_by_id_removes_duplicates_preserving_first():
    d1 = Document(page_content="a", metadata={"id": "1"})
    d2 = Document(page_content="b", metadata={"id": "1"})
    d3 = Document(page_content="c", metadata={"id": "2"})
    assert hybrid._dedupe_by_id([d1, d2, d3]) == [d1, d3]


def test_search_single_repo_calls_retrieve_once(monkeypatch):
    calls = []

    def fake_retrieve(query, *, repo=None, fetch_k=None, settings=None):
        calls.append({"repo": repo, "fetch_k": fetch_k})
        return [Document(page_content="x", metadata={"id": "1", "repo": repo})]

    monkeypatch.setattr(hybrid, "retrieve", fake_retrieve)
    monkeypatch.setattr(hybrid, "rerank", lambda query, docs, *, k, settings: docs)

    result = hybrid.search("pregunta", repo="ws-arg", settings=_settings())

    assert len(calls) == 1
    assert calls[0]["repo"] == "ws-arg"
    assert len(result) == 1


def _record_repo(calls: list) -> callable:
    def fake_retrieve(query, *, repo=None, fetch_k=None, settings=None):
        calls.append(repo)
        return []

    return fake_retrieve


def test_search_none_repo_calls_retrieve_once_with_none(monkeypatch):
    calls: list = []
    monkeypatch.setattr(hybrid, "retrieve", _record_repo(calls))
    monkeypatch.setattr(hybrid, "rerank", lambda query, docs, *, k, settings: docs)

    hybrid.search("pregunta", repo=None, settings=_settings())

    assert calls == [None]


def test_search_single_item_list_behaves_like_single_repo(monkeypatch):
    calls: list = []
    monkeypatch.setattr(hybrid, "retrieve", _record_repo(calls))
    monkeypatch.setattr(hybrid, "rerank", lambda query, docs, *, k, settings: docs)

    hybrid.search("pregunta", repo=["ws-arg"], settings=_settings())

    assert calls == ["ws-arg"]


def test_search_multi_repo_calls_retrieve_and_rerank_per_repo_not_globally(monkeypatch):
    # Regresión del bug real: un solo rerank GLOBAL sobre el pool combinado podía
    # volver a dejar un repo entero fuera del resultado final (truncado por
    # _MAX_CANDIDATES según el orden de concatenación, o por debajo del umbral de
    # relevancia) aunque el retrieval por repo ya hubiera traído candidatos reales
    # para ambos. Rerankear por repo y unir garantiza que los dos sobrevivan.
    retrieve_calls = []
    rerank_calls = []

    def fake_retrieve(query, *, repo=None, fetch_k=None, settings=None):
        retrieve_calls.append({"repo": repo, "fetch_k": fetch_k})
        return [Document(page_content="x", metadata={"id": f"{repo}-1", "repo": repo})]

    def fake_rerank(query, docs, *, k, settings):
        rerank_calls.append({"repo": docs[0].metadata["repo"] if docs else None, "k": k})
        return docs

    monkeypatch.setattr(hybrid, "retrieve", fake_retrieve)
    monkeypatch.setattr(hybrid, "rerank", fake_rerank)

    settings = _settings(
        retrieval_fetch_k=24, retrieval_multi_repo_min_fetch_k=4,
        retrieval_k=8, retrieval_multi_repo_min_k=2,
    )
    result = hybrid.search("pregunta", repo=["ws-arg", "ws-br"], settings=settings)

    assert len(retrieve_calls) == 2
    assert {c["repo"] for c in retrieve_calls} == {"ws-arg", "ws-br"}
    assert all(c["fetch_k"] == 12 for c in retrieve_calls)  # 24 // 2 = 12, por encima del piso de 4

    assert len(rerank_calls) == 2  # un rerank POR repo, no uno global sobre el pool combinado
    assert {c["repo"] for c in rerank_calls} == {"ws-arg", "ws-br"}
    assert all(c["k"] == 4 for c in rerank_calls)  # 8 // 2 = 4

    assert len(result) == 2  # ambos repos sobreviven en el resultado final
    assert {d.metadata["repo"] for d in result} == {"ws-arg", "ws-br"}


def test_search_multi_repo_respects_min_fetch_k_floor(monkeypatch):
    calls = []
    monkeypatch.setattr(
        hybrid,
        "retrieve",
        lambda query, *, repo=None, fetch_k=None, settings=None: calls.append(fetch_k) or [],
    )
    monkeypatch.setattr(hybrid, "rerank", lambda query, docs, *, k, settings: docs)

    settings = _settings(retrieval_fetch_k=24, retrieval_multi_repo_min_fetch_k=4)
    # 24 // 8 == 3, por debajo del piso de 4 -> debe usarse el piso.
    hybrid.search("pregunta", repo=list("abcdefgh"), settings=settings)

    assert all(fk == 4 for fk in calls)


def test_search_multi_repo_dedupes_by_id(monkeypatch):
    shared_doc = Document(page_content="x", metadata={"id": "shared", "repo": "ws-arg"})
    monkeypatch.setattr(
        hybrid, "retrieve", lambda query, *, repo=None, fetch_k=None, settings=None: [shared_doc]
    )
    monkeypatch.setattr(hybrid, "rerank", lambda query, docs, *, k, settings: docs)

    result = hybrid.search("pregunta", repo=["ws-arg", "ws-br"], settings=_settings())

    assert len(result) == 1


def test_search_multi_repo_empty_candidates_for_one_repo_does_not_crash(monkeypatch):
    # Un repo mencionado que no matcheó nada real (nombre inventado/typo) debe
    # simplemente aportar cero docs, no romper la unión con el otro repo.
    def fake_retrieve(query, *, repo=None, fetch_k=None, settings=None):
        if repo == "ws-br":
            return []
        return [Document(page_content="x", metadata={"id": f"{repo}-1", "repo": repo})]

    monkeypatch.setattr(hybrid, "retrieve", fake_retrieve)
    monkeypatch.setattr(hybrid, "rerank", lambda query, docs, *, k, settings: docs)

    result = hybrid.search("pregunta", repo=["ws-arg", "ws-br"], settings=_settings())

    assert len(result) == 1
    assert result[0].metadata["repo"] == "ws-arg"
