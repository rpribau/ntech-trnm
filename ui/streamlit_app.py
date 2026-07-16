"""Frontend v1: chat con el agente de revisión + generación de reportes.

Ejecutar:  streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

# Permite importar el paquete al correr `streamlit run ui/streamlit_app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from config.settings import get_settings  # noqa: E402
from ntech_agent.graph.builder import run_agent  # noqa: E402
from ntech_agent.graph.supervisor import list_local_repos  # noqa: E402
from ui import insights_tab  # noqa: E402

st.set_page_config(page_title="NTech Code Review Agent", page_icon="🔍", layout="wide")
settings = get_settings()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------- Sidebar ------------------------------------- #
with st.sidebar:
    st.header("⚙️ Estado")
    st.caption(f"Backend LLM: **{settings.llm_backend}**")
    st.caption(f"Modelo: `{settings.active_model}`")
    st.caption(f"Embeddings: `{settings.embedding_model}`")

    repos = list_local_repos(settings)
    st.header("📦 Repos indexados")
    if repos:
        st.write(", ".join(f"`{r}`" for r in repos))
    else:
        st.info("No hay repos. Corre `python -m scripts.sync_repos` y `build_index`.")

    st.header("📝 Reportes")
    repo_sel = st.selectbox("Repo", options=repos or ["(ninguno)"])
    col1, col2 = st.columns(2)
    if col1.button("Reporte repo", use_container_width=True) and repos:
        from ntech_agent.report.generate import generate_repo_report

        with st.spinner(f"Generando reporte de {repo_sel}…"):
            path = generate_repo_report(repo_sel)
        st.success(f"Escrito: {path}")
    if col2.button("Reporte org", use_container_width=True) and repos:
        from ntech_agent.report.generate import generate_org_report

        with st.spinner("Generando reporte de la organización…"):
            path = generate_org_report()
        st.success(f"Escrito: {path}")

    if st.button("Nueva conversación", use_container_width=True):
        st.session_state.thread_id = uuid.uuid4().hex
        st.session_state.history = []
        st.rerun()

st.title("🔍 NTech Code Review Agent")

# st.chat_input solo se fija al fondo del viewport cuando se llama a nivel raíz
# del script — anidarlo dentro de un contenedor como st.tabs() rompe ese
# posicionamiento (queda flotando arriba de todo, muy incómodo). Por eso acá se
# usa un radio (widget normal, no contenedor) en vez de tabs, y el chat_input
# se llama sin ningún `with` alrededor.
view = st.radio(
    "Sección", ["💬 Chat", "📊 Insights"], horizontal=True, label_visibility="collapsed"
)

# ----------------------------- Insights ------------------------------------ #
if view == "📊 Insights":
    insights_tab.render(settings, repos)

# ----------------------------- Chat ---------------------------------------- #
else:
    st.caption("Pregunta por un repo (p. ej. *“Dame un resumen de ws-arg”*) o pide una revisión.")

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Escribe tu consulta…"):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Recuperando contexto y razonando…"):
                final = run_agent(prompt, thread_id=st.session_state.thread_id)
            answer = final.get("answer", "_(sin respuesta)_")
            repos_detected = final.get("repos") or ([final["repo"]] if final.get("repo") else [])
            repos_txt = ", ".join(f"`{r}`" for r in repos_detected) or "`(ninguno)`"
            st.caption(f"ruta: `{final.get('route')}`  ·  repos: {repos_txt}")
            st.markdown(answer)
        st.session_state.history.append({"role": "assistant", "content": answer})
