"""Integración con el **GitHub MCP** (remoto/hosted) para consultas en vivo.

Expone las tools del servidor MCP de GitHub (issues, PRs, commits, contenido de
archivos, etc.) como tools de LangChain, para complementar el índice RAG con
datos frescos. Requiere ``langchain-mcp-adapters``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config.settings import Settings, get_settings


def _connections(settings: Settings) -> dict[str, dict[str, Any]]:
    if not settings.github_token:
        raise RuntimeError("NTECH_GITHUB_TOKEN no está configurado en .env")
    return {
        "github": {
            "url": settings.github_mcp_url,
            "transport": "streamable_http",
            "headers": {"Authorization": f"Bearer {settings.github_token}"},
        }
    }


async def get_github_mcp_tools(settings: Settings | None = None) -> list:
    """Devuelve las tools de LangChain expuestas por el GitHub MCP (async)."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    settings = settings or get_settings()
    client = MultiServerMCPClient(_connections(settings))
    return await client.get_tools()


def get_github_mcp_tools_sync(settings: Settings | None = None) -> list:
    """Wrapper síncrono de :func:`get_github_mcp_tools` (para código no-async)."""
    return asyncio.run(get_github_mcp_tools(settings))


if __name__ == "__main__":  # python -m ntech_agent.github.mcp
    tools = get_github_mcp_tools_sync()
    print(f"GitHub MCP expone {len(tools)} tools:")
    for t in tools:
        print(f"  - {t.name}")
