"""Análisis estático: señales objetivas que alimentan al nodo revisor.

Python-first (ruff + radon). El diseño es *pluggable* por lenguaje: si en el
futuro hay repos JS/Go, se añaden runners al dict ``_RUNNERS``.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from config.settings import Settings, get_settings

_TOP_LINT = 15
_TOP_COMPLEX = 12
_TOP_LOW_MI = 10
# Radon: rank C/D/E/F ~ complejidad creciente; MI < 65 (rank B/C) = mantenibilidad baja.
_CC_MIN_RANK = {"C", "D", "E", "F"}


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=300)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"herramienta no encontrada: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def _ruff(repo_dir: Path) -> dict:
    code, out, err = _run(["ruff", "check", ".", "--output-format", "json"], repo_dir)
    if code == 127:
        return {"available": False, "note": err}
    try:
        items = json.loads(out) if out.strip() else []
    except json.JSONDecodeError:
        return {"available": False, "note": "no se pudo parsear la salida de ruff"}

    by_rule = Counter(i.get("code") or "?" for i in items)
    top = [
        {
            "code": i.get("code"),
            "message": i.get("message"),
            "path": i.get("filename"),
            "line": (i.get("location") or {}).get("row"),
        }
        for i in items[:_TOP_LINT]
    ]
    return {
        "available": True,
        "total": len(items),
        "by_rule": dict(by_rule.most_common(15)),
        "top": top,
    }


def _radon_cc(repo_dir: Path) -> dict:
    code, out, err = _run(["radon", "cc", ".", "-j"], repo_dir)
    if code == 127:
        return {"available": False, "note": err}
    try:
        data = json.loads(out) if out.strip() else {}
    except json.JSONDecodeError:
        return {"available": False, "note": "no se pudo parsear radon cc"}

    complexities: list[int] = []
    high: list[dict] = []
    for path, blocks in data.items():
        if isinstance(blocks, dict) and blocks.get("error"):
            continue
        for b in blocks:
            cc = b.get("complexity", 0)
            complexities.append(cc)
            if b.get("rank") in _CC_MIN_RANK:
                high.append(
                    {
                        "name": b.get("name"),
                        "path": path,
                        "line": b.get("lineno"),
                        "complexity": cc,
                        "rank": b.get("rank"),
                    }
                )
    high.sort(key=lambda x: x["complexity"], reverse=True)
    avg = round(sum(complexities) / len(complexities), 2) if complexities else 0.0
    return {"available": True, "avg_cc": avg, "n_blocks": len(complexities),
            "high_complexity": high[:_TOP_COMPLEX]}


def _radon_mi(repo_dir: Path) -> dict:
    code, out, err = _run(["radon", "mi", ".", "-j"], repo_dir)
    if code == 127:
        return {"available": False, "note": err}
    try:
        data = json.loads(out) if out.strip() else {}
    except json.JSONDecodeError:
        return {"available": False, "note": "no se pudo parsear radon mi"}

    mis: list[float] = []
    low: list[dict] = []
    for path, res in data.items():
        if not isinstance(res, dict) or "mi" not in res:
            continue
        mi = res["mi"]
        mis.append(mi)
        if res.get("rank", "A") != "A":  # B/C = mantenibilidad baja
            low.append({"path": path, "mi": round(mi, 1), "rank": res.get("rank")})
    low.sort(key=lambda x: x["mi"])
    avg = round(sum(mis) / len(mis), 1) if mis else 0.0
    return {"available": True, "avg_mi": avg, "n_files": len(mis), "low_mi_files": low[:_TOP_LOW_MI]}


def _python_runner(repo_dir: Path) -> dict:
    return {
        "lint": _ruff(repo_dir),
        "complexity": _radon_cc(repo_dir),
        "maintainability": _radon_mi(repo_dir),
    }


_RUNNERS = {"python": _python_runner}


def run_static_analysis(repo_dir: Path, settings: Settings | None = None) -> dict:
    """Corre el análisis estático sobre un repo clonado. Devuelve un dict estructurado."""
    settings = settings or get_settings()
    if not settings.static_analysis_enabled:
        return {"enabled": False}
    if not repo_dir.exists():
        return {"enabled": True, "error": f"no existe {repo_dir}"}

    has_python = any(repo_dir.rglob("*.py"))
    result: dict = {"enabled": True, "repo": repo_dir.name, "languages": []}
    if has_python:
        result["languages"].append("python")
        result["python"] = _RUNNERS["python"](repo_dir)
    return result


def format_for_prompt(findings: dict) -> str:
    """Resumen compacto del análisis estático para inyectar en el prompt del revisor."""
    if not findings.get("enabled"):
        return "Análisis estático desactivado."
    if findings.get("error"):
        return f"Análisis estático no disponible: {findings['error']}"

    py = findings.get("python")
    if not py:
        return "Sin archivos Python analizables (o lenguaje no soportado aún)."

    lines: list[str] = []
    lint = py.get("lint", {})
    if lint.get("available"):
        lines.append(f"Lint (ruff): {lint['total']} hallazgos. Top reglas: {lint['by_rule']}")
        for t in lint.get("top", [])[:8]:
            lines.append(f"  - {t['code']} {t['path']}:{t['line']} — {t['message']}")
    else:
        lines.append(f"Lint (ruff): no disponible ({lint.get('note', '')}).")

    cc = py.get("complexity", {})
    if cc.get("available"):
        lines.append(f"Complejidad (radon cc): promedio {cc['avg_cc']} en {cc['n_blocks']} bloques.")
        for h in cc.get("high_complexity", [])[:6]:
            lines.append(f"  - {h['rank']} cc={h['complexity']} {h['path']}:{h['line']} {h['name']}")

    mi = py.get("maintainability", {})
    if mi.get("available"):
        lines.append(f"Mantenibilidad (radon mi): promedio {mi['avg_mi']} en {mi['n_files']} archivos.")
        for m in mi.get("low_mi_files", [])[:6]:
            lines.append(f"  - MI={m['mi']} ({m['rank']}) {m['path']}")

    return "\n".join(lines) if lines else "Sin señales de análisis estático."
