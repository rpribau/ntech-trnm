"""Extracción de símbolos (definiciones) y referencias por lenguaje, vía tree-sitter.

Diseño pluggable por lenguaje, igual que ``ntech_agent/analysis/static.py``: un
dict ``_LANG_EXTRACTORS`` por lenguaje soportado; lenguajes sin entrada degradan
con gracia (el archivo queda sin símbolos, disponible igual vía RAG normal).

Solo extraemos **definiciones** vía AST (funciones/métodos/clases + su rango de
líneas). Las **referencias** (para armar el grafo de dependencias) se cuentan por
texto (regex de límite de palabra) contra el set global de nombres definidos en el
repo — una aproximación consciente frente a un sistema completo de tags AST
(referencias son mucho más frágiles de mapear correctamente entre gramáticas,
sobre todo en C++), documentada como simplificación de alcance de tesis.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tree_sitter_language_pack import get_parser

_EXT_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".ipynb": "python",  # ver notebook_code_to_text: se preprocesa antes de parsear
    ".js": "javascript",
    ".jsx": "javascript",
    ".c": "cpp",
    ".h": "cpp",
    ".cpp": "cpp",
    ".cc": "cpp",
}

# Nombres cortos/comunes que generan demasiados falsos positivos en el conteo de
# referencias por texto (colisión entre símbolos no relacionados de distintos archivos).
_REF_STOPWORDS = {
    "main", "run", "init", "get", "set", "update", "process", "start", "stop",
    "close", "open", "load", "save", "render", "build", "create", "delete",
    "handle", "value", "data", "self", "this", "id", "type", "name", "index",
    "count", "size", "length", "print", "log", "test", "app", "config",
}
_REF_MIN_LEN = 3


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str  # class | struct | function | method
    start_line: int
    end_line: int


def language_for_path(path: Path) -> str | None:
    return _EXT_LANGUAGE.get(path.suffix.lower())


def is_notebook(path: Path) -> bool:
    return path.suffix.lower() == ".ipynb"


def notebook_code_to_text(raw: str) -> str:
    """Concatena solo las celdas de código de un .ipynb (se ignoran las celdas
    markdown) como si fuera un único archivo Python — permite reusar el mismo
    extractor de Python (tree-sitter) y mostrarle al reviewer código legible en
    vez del JSON crudo del notebook. Asume kernel Python (igual que el resto del
    proyecto — ver ``ntech_agent/analysis/static.py``, Python-first).

    Los números de línea resultantes no corresponden a líneas reales del .ipynb
    (que es un único blob JSON) — son solo una referencia interna consistente
    para el grafo de símbolos, no una posición navegable en el archivo.
    """
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    parts: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        text = "".join(src) if isinstance(src, list) else str(src)
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _first_child_text(node, types: set[str], src: bytes) -> str | None:
    for c in node.children:
        if c.type in types:
            return src[c.start_byte : c.end_byte].decode("utf-8", "ignore")
    return None


def _span(node) -> tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1


def _walk(root, on_node) -> None:
    """DFS del árbol vía ``TreeCursor`` (el patrón seguro de tree-sitter para
    recorrer árboles enteros). No usar una pila propia de objetos ``Node`` para
    el recorrido de árbol completo: retener muchos ``Node`` sueltos simultáneos
    corrompe memoria de forma silenciosa con los bindings actuales — confirmado
    con un segfault reproducible (y valores de línea con basura antes de eso)
    contra archivos reales de ws-mex/ws-br. ``.children`` sigue siendo seguro
    para búsquedas acotadas e inmediatas dentro de ``on_node`` (ver
    ``_first_child_text``/``_find_function_declarator``).

    ``on_node(node) -> bool``: True para seguir bajando a los hijos de ``node``.
    """
    cursor = root.walk()
    reached_root = False
    while not reached_root:
        descend = on_node(cursor.node)
        if descend and cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue
        retracting = True
        while retracting:
            if not cursor.goto_parent():
                retracting = False
                reached_root = True
            elif cursor.goto_next_sibling():
                retracting = False


def _extract_python(src: bytes) -> list[Symbol]:
    tree = get_parser("python").parse(src)
    out: list[Symbol] = []

    def on_node(node) -> bool:
        if node.type == "function_definition":
            start, end = _span(node)
            name = _first_child_text(node, {"identifier"}, src)
            if name:
                out.append(Symbol(name, "function", start, end))
            return False  # no bajar al cuerpo: evita closures/funciones anidadas
        if node.type == "class_definition":
            start, end = _span(node)
            name = _first_child_text(node, {"identifier"}, src)
            if name:
                out.append(Symbol(name, "class", start, end))
        return True

    _walk(tree.root_node, on_node)
    return out


def _extract_javascript(src: bytes) -> list[Symbol]:
    tree = get_parser("javascript").parse(src)
    out: list[Symbol] = []

    def on_node(node) -> bool:
        if node.type == "function_declaration":
            start, end = _span(node)
            name = _first_child_text(node, {"identifier"}, src)
            if name:
                out.append(Symbol(name, "function", start, end))
            return False
        if node.type == "method_definition":
            start, end = _span(node)
            name = _first_child_text(node, {"property_identifier"}, src)
            if name:
                out.append(Symbol(name, "method", start, end))
            return False
        if node.type == "variable_declarator":
            is_func_value = any(
                c.type in ("arrow_function", "function", "function_expression")
                for c in node.children
            )
            if is_func_value:
                start, end = _span(node)
                name = _first_child_text(node, {"identifier"}, src)
                if name:
                    out.append(Symbol(name, "function", start, end))
                return False  # no bajar al cuerpo del arrow function (componentes React)
        if node.type == "class_declaration":
            start, end = _span(node)
            name = _first_child_text(node, {"identifier"}, src)
            if name:
                out.append(Symbol(name, "class", start, end))
        return True

    _walk(tree.root_node, on_node)
    return out


def _find_function_declarator(node):
    """Busca un ``function_declarator`` en el subárbol de ``node`` (acotado —
    declaradores de C++ rara vez bajan más de unos pocos niveles)."""
    for c in node.children:
        if c.type == "function_declarator":
            return c
        found = _find_function_declarator(c)
        if found is not None:
            return found
    return None


def _declarator_name(declarator, src: bytes) -> str | None:
    for c in declarator.children:
        if c.type == "qualified_identifier":
            idents = [g for g in c.children if g.type in ("identifier", "field_identifier")]
            if idents:
                last = idents[-1]
                return src[last.start_byte : last.end_byte].decode("utf-8", "ignore")
        if c.type in ("identifier", "field_identifier", "destructor_name"):
            return src[c.start_byte : c.end_byte].decode("utf-8", "ignore")
    return None


def _extract_cpp(src: bytes) -> list[Symbol]:
    tree = get_parser("cpp").parse(src)
    out: list[Symbol] = []

    def on_node(node) -> bool:
        if node.type in ("class_specifier", "struct_specifier"):
            start, end = _span(node)
            name = _first_child_text(node, {"type_identifier"}, src)
            if name:
                kind = "class" if node.type == "class_specifier" else "struct"
                out.append(Symbol(name, kind, start, end))
            return True  # seguir bajando: buscar métodos dentro del cuerpo
        if node.type == "function_definition":
            start, end = _span(node)
            declarator = _find_function_declarator(node)
            name = _declarator_name(declarator, src) if declarator is not None else None
            if name:
                out.append(Symbol(name, "function", start, end))
            return False  # no bajar al cuerpo
        if node.type == "field_declaration":
            start, end = _span(node)
            declarator = _find_function_declarator(node)
            if declarator is not None:
                name = _declarator_name(declarator, src)
                if name:
                    out.append(Symbol(name, "method", start, end))
            return False
        return True

    _walk(tree.root_node, on_node)
    return out


_LANG_EXTRACTORS = {
    "python": _extract_python,
    "javascript": _extract_javascript,
    "cpp": _extract_cpp,
}


def extract_definitions(text: str, language: str) -> list[Symbol]:
    """Definiciones (clases/funciones/métodos) de un archivo, vía AST tree-sitter."""
    extractor = _LANG_EXTRACTORS.get(language)
    if extractor is None:
        return []
    return extractor(text.encode("utf-8", "ignore"))


def extract_references(text: str, candidate_names: set[str]) -> set[str]:
    """Identificadores de ``candidate_names`` que aparecen en ``text`` (por texto,
    con límite de palabra completa) — aproximación a "qué archivo referencia qué
    símbolo", sin resolución semántica real.
    """
    names = sorted(
        n for n in candidate_names if len(n) >= _REF_MIN_LEN and n not in _REF_STOPWORDS
    )
    if not names:
        return set()
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")
    return set(pattern.findall(text))
