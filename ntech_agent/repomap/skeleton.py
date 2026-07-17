"""Índice de solo firmas (sin cuerpo) de todos los archivos con símbolos de un
repo — complementa al top-N por PageRank de ``graph.py``: cobertura estructural
del 100% del repo, sin depender de ranking/popularidad. Puro render sobre la
forma de datos que ``build.py`` ya persiste (``files[i]["symbols"]``); no vuelve
a tocar tree-sitter.
"""

from __future__ import annotations

from config.settings import Settings, get_settings

_KIND_LABEL = {"class": "class", "struct": "struct", "function": "def", "method": "def"}


def _nest_symbols(symbols: list[dict]) -> list[dict]:
    """Agrega profundidad de anidamiento (``_depth``) por CONTENCIÓN de rango de
    línea: un símbolo cuyo ``start_line`` cae dentro de un ``class``/``struct``
    anterior aún no cerrado se considera miembro suyo y se renderiza indentado.
    Heurística de post-procesamiento — ``Symbol`` (ver ``tags.py``) no trae
    relación padre-hijo explícita, solo ``name``/``kind``/``start_line``/``end_line``;
    en Python en particular esto es lo único que permite distinguir visualmente
    un método de una función top-level, ya que ``extract_definitions`` emite
    ``kind="function"`` para ambos.
    """
    ordered = sorted(symbols, key=lambda s: s["start_line"])
    stack: list[dict] = []  # contenedores class/struct abiertos, por end_line
    nested: list[dict] = []
    for sym in ordered:
        while stack and stack[-1]["end_line"] < sym["start_line"]:
            stack.pop()
        nested.append({**sym, "_depth": len(stack)})
        if sym["kind"] in ("class", "struct"):
            stack.append(sym)
    return nested


def _render_file_block(file_meta: dict, settings: Settings) -> str | None:
    """``None`` si el archivo no tiene símbolos (se omite del índice)."""
    symbols = file_meta.get("symbols") or []
    if not symbols:
        return None
    nested = _nest_symbols(symbols)
    max_symbols = settings.repomap_skeleton_max_symbols_per_file
    shown = nested[:max_symbols]
    lines = [f"### {file_meta['path']} ({file_meta.get('language', '?')})"]
    for sym in shown:
        indent = "    " * sym["_depth"]
        label = _KIND_LABEL.get(sym["kind"], sym["kind"])
        lines.append(f"{indent}{label} {sym['name']}  [L{sym['start_line']}-{sym['end_line']}]")
    remaining = len(nested) - len(shown)
    if remaining > 0:
        lines.append(f"    (+{remaining} símbolos más)")
    return "\n".join(lines)


def render_repo_skeleton(data: dict, settings: Settings | None = None) -> str:
    """Índice de firmas de TODOS los archivos con símbolos del repo. ``""`` si
    no hay ninguno. Trunca de forma graciosa (nunca omite el índice entero) por
    ``repomap_skeleton_max_files`` — ``data["files"]`` ya viene ordenado
    alfabéticamente desde ``build_repo_map`` — y por ``repomap_skeleton_max_chars``
    (presupuesto total del bloque renderizado, mismo espíritu que
    ``repomap_file_chars`` pero para el índice completo en vez de un solo archivo).
    """
    settings = settings or get_settings()
    eligible = [f for f in data.get("files", []) if f.get("symbols")]
    if not eligible:
        return ""

    capped = eligible[: settings.repomap_skeleton_max_files]
    omitted_by_file_cap = len(eligible) - len(capped)

    blocks: list[str] = []
    total_len = 0
    omitted_by_char_cap = 0
    for i, f in enumerate(capped):
        block = _render_file_block(f, settings)
        if block is None:
            continue
        sep = 2 if blocks else 0
        if total_len + sep + len(block) > settings.repomap_skeleton_max_chars:
            omitted_by_char_cap = len(capped) - i
            break
        blocks.append(block)
        total_len += sep + len(block)

    text = "\n\n".join(blocks)
    omitted = omitted_by_file_cap + omitted_by_char_cap
    if omitted > 0:
        text += f"\n\n### (+{omitted} archivos más, omitidos por límite de tamaño del índice)"
    return text
