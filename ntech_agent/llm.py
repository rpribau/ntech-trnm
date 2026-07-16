"""Cliente LLM: tres backends intercambiables vía ``NTECH_LLM_BACKEND``.

- ``cloudrun``  — vLLM propio en GCP Cloud Run (self-hosted, requiere GPU L4).
- ``ollama``    — modelo local vía Ollama (barato/offline, tu hardware).
- ``anthropic`` — API de Anthropic (Claude), pay-per-token, sin infra que mantener.

``cloudrun`` y ``ollama`` exponen la API compatible con OpenAI (``ChatOpenAI``);
``anthropic`` usa su propio cliente (``ChatAnthropic``) pero implementa la misma
interfaz de LangChain (``invoke``, ``with_structured_output``, ...), así que el
resto del código (grafo, retrieval, rerank) no necesita saber cuál está activo.

Para Cloud Run (servicio privado con IAM) se mintea un **ID token** y se envía
como ``Authorization: Bearer <token>`` (el cliente de OpenAI usa ``api_key`` como
bearer). El token se cachea con TTL para evitar mintarlo en cada llamada.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config.settings import Settings, get_settings

# ---- Caché simple de ID token (audience -> (token, expiry_epoch)) ----
_TOKEN_TTL_SECONDS = 45 * 60  # refrescamos antes de la expiración real (~60 min)


@dataclass
class _CachedToken:
    token: str
    expires_at: float


_token_cache: dict[str, _CachedToken] = {}


def _fetch_id_token(audience: str, settings: Settings) -> str:
    """Obtiene un ID token de GCP para el ``audience`` (URL del servicio Cloud Run).

    Estrategias, en orden:
      1. Service account JSON (``GOOGLE_APPLICATION_CREDENTIALS``).
      2. Application Default Credentials / metadata server.
      3. ``gcloud auth print-identity-token`` como último recurso.
    """
    from google.auth.transport.requests import Request

    # 1) Service account key file.
    if settings.google_application_credentials:
        from google.oauth2 import service_account

        creds = service_account.IDTokenCredentials.from_service_account_file(
            settings.google_application_credentials, target_audience=audience
        )
        creds.refresh(Request())
        return creds.token

    # 2) ADC / metadata server.
    try:
        from google.oauth2 import id_token as google_id_token

        return google_id_token.fetch_id_token(Request(), audience)
    except Exception:
        pass

    # 3) gcloud CLI.
    try:
        out = subprocess.run(
            ["gcloud", "auth", "print-identity-token"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover - depende del entorno
        raise RuntimeError(
            "No se pudo obtener un ID token de GCP. Configura "
            "GOOGLE_APPLICATION_CREDENTIALS o corre `gcloud auth login`."
        ) from exc


def _cloudrun_token(settings: Settings) -> str:
    audience = settings.cloudrun_url.rstrip("/")
    cached = _token_cache.get(audience)
    now = time.time()
    if cached and cached.expires_at > now:
        return cached.token
    token = _fetch_id_token(audience, settings)
    _token_cache[audience] = _CachedToken(token=token, expires_at=now + _TOKEN_TTL_SECONDS)
    return token


def get_chat_model(
    *,
    temperature: float = 0.1,
    max_tokens: int | None = 2048,
    streaming: bool = False,
    settings: Settings | None = None,
) -> BaseChatModel:
    """Devuelve el chat model de LangChain configurado para el backend activo.

    Nota: se construye por-llamada para que el ID token de Cloud Run siempre sea
    fresco en sesiones largas.
    """
    settings = settings or get_settings()

    if settings.llm_backend == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("NTECH_ANTHROPIC_API_KEY no está configurado en .env")
        from langchain_anthropic import ChatAnthropic

        # Opus 4.6+/Sonnet 5/Fable 5 ya no aceptan `temperature` (400 si se envía):
        # el sampling se controla con prompting, no con este parámetro. Se omite
        # aquí para todos los modelos Anthropic en vez de mantener una lista de
        # excepciones por modelo.
        kwargs: dict = dict(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            max_tokens=max_tokens,
            streaming=streaming,
            timeout=600,
            max_retries=2,
        )
        # Desactivar thinking explícitamente: en Sonnet 5 (y otros modelos 4.6+)
        # el thinking adaptativo viene ENCENDIDO por defecto si se omite el
        # parámetro, lo cual (a) puede agotar max_tokens solo pensando, dejando
        # la respuesta vacía, y (b) rompe el structured output (Anthropic exige
        # tool_choice="auto" con thinking activo, y with_structured_output puede
        # forzar una tool). No necesitamos razonamiento extendido para
        # routing/rerank/reportes. Fable 5 / Mythos 5 son la excepción: rechazan
        # el "disabled" explícito (400) porque en esos modelos el thinking
        # siempre está encendido.
        model_lower = settings.anthropic_model.lower()
        if "fable" not in model_lower and "mythos" not in model_lower:
            kwargs["thinking"] = {"type": "disabled"}
        return ChatAnthropic(**kwargs)

    if settings.llm_backend == "cloudrun":
        if not settings.cloudrun_url:
            raise RuntimeError("NTECH_CLOUDRUN_URL no está configurado en .env")
        base_url = settings.cloudrun_url.rstrip("/") + "/v1"
        api_key = _cloudrun_token(settings)  # se envía como Bearer -> IAM de Cloud Run
        model = settings.cloudrun_model
    else:  # ollama
        base_url = settings.ollama_url  # ya incluye /v1
        api_key = "ollama"  # dummy; Ollama no valida
        model = settings.ollama_model

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        timeout=600,  # tolera cold-starts de Cloud Run
        max_retries=2,
    )


def extract_text(content: object) -> str:
    """Extrae el texto de una respuesta de chat, sea ``str`` o lista de bloques.

    Algunos modelos (p. ej. Claude con thinking activo) devuelven ``.content``
    como una lista de bloques (``{"type": "thinking", ...}``, ``{"type": "text",
    ...}``). Usar la lista directamente en un f-string imprime su ``repr()``
    crudo; esta función se queda solo con el texto legible.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        text = "\n".join(p for p in parts if p)
        return text or "_(el modelo no devolvió texto — revisa max_tokens/thinking)_"
    return str(content)


def ping() -> dict:
    """Smoke test: pide una respuesta mínima al modelo activo."""
    settings = get_settings()
    llm = get_chat_model(max_tokens=16, settings=settings)
    resp = llm.invoke("Responde solo con: OK")
    return {
        "backend": settings.llm_backend,
        "model": settings.active_model,
        "reply": extract_text(resp.content),
    }


if __name__ == "__main__":  # python -m ntech_agent.llm
    print(ping())
