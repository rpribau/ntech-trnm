from __future__ import annotations

from types import SimpleNamespace

from config.settings import Settings
from ntech_agent.report.executive import (
    ExecutiveReport,
    _NarrativaEjecutiva,
    _org_evidence,
    _template_actividad_desarrollo,
    _template_introduccion,
    _template_metodologia,
    _template_objetivos,
    build_org_executive_report,
    build_repo_executive_report,
    render_executive_report,
)


class _FakeStructuredLLM:
    def __init__(self, parent, result):
        self._parent = parent
        self._result = result

    def invoke(self, prompt):
        self._parent.last_prompt = prompt
        return self._result


class _FakeLLM:
    def __init__(
        self,
        structured_result=None,
        raise_on_structured=False,
        plain_text="",
        raise_on_invoke=False,
    ):
        self._structured_result = structured_result
        self._raise_on_structured = raise_on_structured
        self._plain_text = plain_text
        self._raise_on_invoke = raise_on_invoke
        self.last_prompt = None

    def with_structured_output(self, model):
        if self._raise_on_structured:
            raise RuntimeError("structured output no soportado")
        return _FakeStructuredLLM(self, self._structured_result)

    def invoke(self, prompt):
        self.last_prompt = prompt
        if self._raise_on_invoke:
            raise RuntimeError("backend caído")
        return SimpleNamespace(content=self._plain_text)


def test_template_objetivos_mentions_repo():
    assert "`ws-br`" in _template_objetivos("ws-br")
    assert "organización" in _template_objetivos(None)


def test_template_metodologia_mentions_repomap_and_static_langs():
    text = _template_metodologia(used_repomap=True, static_languages=["python"])
    assert "mapa estructural" in text
    assert "python" in text

    text_no_repomap = _template_metodologia(used_repomap=False, static_languages=[])
    assert "mapa estructural" not in text_no_repomap


def test_template_introduccion_includes_resumen():
    text = _template_introduccion("ws-br", "Es un sistema de conteo vehicular.")
    assert "ws-br" in text
    assert "conteo vehicular" in text


def test_template_actividad_desarrollo_repo_no_data():
    text = _template_actividad_desarrollo("ws-br", None)
    assert "no hay commits sincronizados" in text.lower()


def test_template_actividad_desarrollo_repo_with_data():
    facts = {
        "total_commits": 5,
        "contributors": [{"author": "rpribau", "commits": 5}],
        "last_commit": {
            "committed_at": "2026-07-10T12:00:00+00:00",
            "author": "rpribau",
            "commit_message": "fix",
        },
    }
    text = _template_actividad_desarrollo("ws-br", facts)
    assert "5 commits" in text
    assert "2026-07-10" in text
    assert "rpribau" in text


def test_template_actividad_desarrollo_org_no_data():
    text = _template_actividad_desarrollo(None, {"ws-br": {"total_commits": 0}})
    assert "no hay commits sincronizados" in text.lower()


def test_template_actividad_desarrollo_org_aggregates_repos():
    facts = {
        "ws-br": {
            "total_commits": 3,
            "contributors": [{"author": "a", "commits": 3}],
            "last_commit": {"committed_at": "2026-07-01T00:00:00+00:00"},
        },
        "ws-mex": {
            "total_commits": 2,
            "contributors": [{"author": "b", "commits": 2}],
            "last_commit": {"committed_at": "2026-07-05T00:00:00+00:00"},
        },
    }
    text = _template_actividad_desarrollo(None, facts)
    assert "5 commits" in text  # 3 + 2 agregados
    assert "`ws-br`" in text
    assert "`ws-mex`" in text


def test_build_repo_executive_report_uses_structured_narrativa(monkeypatch):
    fake_result = _NarrativaEjecutiva(
        principales_hallazgos_y_conclusiones="El repo está en buen estado general.",
        recomendaciones="Priorizar pruebas automatizadas.",
    )
    monkeypatch.setattr(
        "ntech_agent.report.executive.get_chat_model",
        lambda **kwargs: _FakeLLM(structured_result=fake_result),
    )
    settings = Settings(_env_file=None)

    review_data = {
        "resumen_contextualizado": "Sistema de conteo vehicular con visión por computadora.",
        "fortalezas": ["Buena separación de módulos"],
        "areas_de_oportunidad": ["Configuración hardcodeada"],
        "hallazgos": [
            {
                "ubicacion": "cpp-motor/src/VehicleCounter.cpp:56",
                "categoria": "Anti-patterns",
                "severidad": "mayor",
                "hallazgo": "Coordenadas hardcodeadas",
                "por_que": "Requiere recompilar para calibrar",
                "guia": "guidelines/software_dev/anti_patterns_code_smells.md",
                "recomendacion": "Externalizar a config",
            }
        ],
    }

    report = build_repo_executive_report(
        "ws-br",
        review_data=review_data,
        review_markdown="(markdown crudo, no debería usarse aquí)",
        static_findings={"languages": ["python"]},
        used_repomap=True,
        settings=settings,
    )

    assert isinstance(report, ExecutiveReport)
    assert report.repo == "ws-br"
    assert report.principales_hallazgos_y_conclusiones == "El repo está en buen estado general."
    assert report.recomendaciones == "Priorizar pruebas automatizadas."
    assert "VehicleCounter.cpp" not in report.introduccion  # sin jerga de rutas de archivo
    assert "conteo vehicular" in report.introduccion


def test_build_repo_executive_report_falls_back_to_plain_text(monkeypatch):
    monkeypatch.setattr(
        "ntech_agent.report.executive.get_chat_model",
        lambda **kwargs: _FakeLLM(raise_on_structured=True, plain_text="Texto libre de respaldo."),
    )
    settings = Settings(_env_file=None)

    report = build_repo_executive_report(
        "ws-arg",
        review_data={},  # cayó al fallback de texto libre en reviewer_node
        review_markdown="**Resumen:** algo\n\n**Hallazgos:**\n- [mayor] ...",
        static_findings={},
        used_repomap=False,
        settings=settings,
    )

    assert report.principales_hallazgos_y_conclusiones == "Texto libre de respaldo."
    assert report.recomendaciones == ""


def test_build_repo_executive_report_degrades_gracefully_when_both_llm_calls_fail(monkeypatch):
    # Antes de este fix, si el fallback de texto libre TAMBIÉN fallaba (p. ej. un
    # 5xx transitorio del proveedor), la excepción se propagaba sin control.
    monkeypatch.setattr(
        "ntech_agent.report.executive.get_chat_model",
        lambda **kwargs: _FakeLLM(raise_on_structured=True, raise_on_invoke=True),
    )
    settings = Settings(_env_file=None)

    report = build_repo_executive_report(
        "ws-arg",
        review_data={},
        review_markdown="**Resumen:** algo",
        static_findings={},
        used_repomap=False,
        settings=settings,
    )

    assert "no respondió" in report.principales_hallazgos_y_conclusiones
    assert report.recomendaciones == ""


def test_render_executive_report_has_exactly_six_sections_in_order():
    report = ExecutiveReport(
        repo="ws-mex",
        introduccion="intro",
        objetivos_generales="objetivos",
        metodologia="metodologia",
        actividad_desarrollo="actividad",
        principales_hallazgos_y_conclusiones="hallazgos",
        recomendaciones="recomendaciones",
    )
    md = render_executive_report(report)
    headers = [line for line in md.splitlines() if line.startswith("## ")]
    assert headers == [
        "## Introducción",
        "## Objetivos generales",
        "## Metodología",
        "## Actividad de desarrollo",
        "## Principales hallazgos y conclusiones",
        "## Recomendaciones",
    ]
    assert md.startswith("# Reporte ejecutivo — ws-mex")


def test_render_executive_report_org_title_has_no_repo():
    report = ExecutiveReport(
        repo=None,
        introduccion="i",
        objetivos_generales="o",
        metodologia="m",
        actividad_desarrollo="a",
        principales_hallazgos_y_conclusiones="h",
        recomendaciones="r",
    )
    md = render_executive_report(report)
    assert md.startswith("# Reporte ejecutivo — Organización NTech-TRNM")


def test_build_org_executive_report_aggregates_multiple_repos(monkeypatch):
    fake_result = _NarrativaEjecutiva(
        principales_hallazgos_y_conclusiones="Panorama consolidado.",
        recomendaciones="Invertir en testing en toda la org.",
    )
    monkeypatch.setattr(
        "ntech_agent.report.executive.get_chat_model",
        lambda **kwargs: _FakeLLM(structured_result=fake_result),
    )
    settings = Settings(_env_file=None)

    per_repo = {
        "ws-br": {
            "resumen_contextualizado": "...",
            "hallazgos": [{"severidad": "mayor"}],
            "areas_de_oportunidad": [],
        },
        "ws-mex": {},
    }
    report = build_org_executive_report(
        per_repo, {"ws-br": ["cpp"], "ws-mex": ["python"]}, used_repomap=True, settings=settings
    )

    assert report.repo is None
    assert "ws-br" in report.introduccion
    assert "ws-mex" in report.introduccion
    assert "2 repositorios" in report.introduccion


def test_org_evidence_includes_languages_by_repo():
    per_repo = {"ws-br": {"resumen_contextualizado": "..."}, "ws-arg": {}}
    evidence = _org_evidence(per_repo, {"ws-br": ["cpp", "python"], "ws-arg": ["python"]})
    assert "cpp, python" in evidence
    assert "### ws-arg\nLenguajes: python" in evidence


def test_org_evidence_missing_languages_shows_placeholder():
    evidence = _org_evidence({"ws-mex": {}}, {})
    assert "no detectado" in evidence


def test_build_org_executive_report_uses_org_prompt_asking_cross_repo_patterns(monkeypatch):
    fake_result = _NarrativaEjecutiva(
        principales_hallazgos_y_conclusiones="Panorama.", recomendaciones="Acciones."
    )
    created: list[_FakeLLM] = []

    def factory(**kwargs):
        llm = _FakeLLM(structured_result=fake_result)
        created.append(llm)
        return llm

    monkeypatch.setattr("ntech_agent.report.executive.get_chat_model", factory)
    settings = Settings(_env_file=None)

    build_org_executive_report(
        {"ws-br": {}, "ws-mex": {}}, {"ws-br": ["cpp"], "ws-mex": ["python"]},
        used_repomap=False, settings=settings,
    )

    assert "PATRONES QUE SE REPITEN" in created[-1].last_prompt
    assert "organización" in created[-1].last_prompt.lower()


def test_build_repo_executive_report_still_uses_repo_prompt(monkeypatch):
    fake_result = _NarrativaEjecutiva(
        principales_hallazgos_y_conclusiones="Resumen.", recomendaciones="Acciones."
    )
    created: list[_FakeLLM] = []

    def factory(**kwargs):
        llm = _FakeLLM(structured_result=fake_result)
        created.append(llm)
        return llm

    monkeypatch.setattr("ntech_agent.report.executive.get_chat_model", factory)
    settings = Settings(_env_file=None)

    build_repo_executive_report(
        "ws-br", review_data={}, review_markdown="", static_findings={},
        used_repomap=False, settings=settings,
    )

    assert "PATRONES QUE SE REPITEN" not in created[-1].last_prompt
