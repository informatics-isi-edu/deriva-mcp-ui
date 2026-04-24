"""FastAPI application: route registration and lifespan."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import pathlib
import time
from urllib.parse import urlparse
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .audit import audit_event, init_audit_logger
from .auth import (
    ANON_COOKIE_NAME,
    RequireSession,
    _extract_display_name,
    history_key,
    user_session_key,
)
from .auth import router as auth_router
from .chat import ChatCancelled, run_chat_turn
from .config import Settings
from .mcp_client import MCPAuthError
from .storage import Session, create_store

logger = logging.getLogger(__name__)

class _RagModeBody(BaseModel):
    enabled: bool

def _init_logging(debug: bool = False, app_use_syslog: bool = False) -> None:  # pragma: no cover
    """Configure the root deriva_mcp_ui logger.

    Always adds a stderr StreamHandler (for ``docker logs`` and local dev).
    Optionally adds a SysLogHandler on LOCAL1 when *app_use_syslog* is True,
    for non-Docker deployments where syslog is the only path to a centralized
    collector.  In Docker, ``driver: syslog`` in compose already forwards
    stderr, so enabling this would duplicate every app log line.

    Audit and access logs have their own SysLogHandlers (LOCAL1/LOCAL2)
    controlled by separate config flags.
    """
    import os

    fmt_stream = logging.Formatter(
        "%(asctime)s [%(process)d] [%(levelname)s] [%(name)s] - %(message)s"
    )

    root = logging.getLogger("deriva_mcp_ui")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt_stream)
    root.addHandler(stream_handler)

    if app_use_syslog:
        syslog_socket = "/dev/log"
        if os.path.exists(syslog_socket) and os.access(syslog_socket, os.W_OK):
            from logging.handlers import SysLogHandler

            try:
                sh = SysLogHandler(address=syslog_socket, facility=SysLogHandler.LOG_LOCAL1)
                sh.ident = "deriva-mcp-ui: "
                sh.setFormatter(logging.Formatter(
                    "[%(process)d] [%(levelname)s] [%(name)s] - %(message)s"
                ))
                root.addHandler(sh)
            except Exception:
                pass

    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.propagate = False

    # Suppress per-request noise from httpx/httpcore
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Route the uvicorn logger through our stream handler so startup messages
    # appear in stderr without propagating to the root logger.
    uv_log = logging.getLogger("uvicorn")
    uv_log.handlers = []
    uv_log.addHandler(stream_handler)
    uv_log.propagate = False

    # Detach uvicorn.access from the main log stream.  Handlers are added
    # later by _init_access_logging() once settings are available.
    access_log = logging.getLogger("uvicorn.access")
    access_log.handlers = []
    access_log.propagate = False
    access_log.setLevel(logging.INFO)


class _HealthCheckThrottle(logging.Filter):  # pragma: no cover
    _interval = 600.0
    _last_logged: float = 0.0

    def filter(self, record: logging.LogRecord) -> bool:
        if "/health" not in record.getMessage():
            return True
        now = time.monotonic()
        if now - self.__class__._last_logged >= self.__class__._interval:
            self.__class__._last_logged = now
            return True
        return False


def _init_access_logging(settings: Settings) -> None:  # pragma: no cover
    """Route uvicorn access logs to syslog (LOCAL2) or stderr.

    Uses LOG_LOCAL2 so rsyslog can route access logs separately from the
    application log (LOCAL1) and audit log (LOCAL1 with dedicated ident).
    Falls back to stderr when syslog is unavailable or not requested, so
    Docker / AWS log drivers capture access lines alongside app output.
    """
    import os
    from logging.handlers import SysLogHandler

    access_log = logging.getLogger("uvicorn.access")
    access_log.addFilter(_HealthCheckThrottle())

    if settings.access_use_syslog:
        syslog_socket = "/dev/log"
        if os.path.exists(syslog_socket) and os.access(syslog_socket, os.W_OK):
            try:
                sh = SysLogHandler(
                    address=syslog_socket, facility=SysLogHandler.LOG_LOCAL2
                )
                sh.ident = "deriva-mcp-ui-access: "
                sh.setFormatter(logging.Formatter("%(message)s"))
                access_log.addHandler(sh)
                return
            except Exception:
                pass

    # Fallback: stderr (picked up by Docker / AWS log drivers)
    fh = logging.StreamHandler()
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    access_log.addHandler(fh)


class ChatRequest(BaseModel):
    message: str
    hostname: str = ""
    catalog_id: str = ""
    session_id: str = ""  # client echoes back the session_id it received at page load


@asynccontextmanager
async def _lifespan(app: FastAPI):  # pragma: no cover
    settings: Settings = app.state.settings
    store = create_store(
        settings.storage_backend,
        settings.storage_backend_url,
        settings.session_ttl,
    )
    app.state.store = store

    async def _sweep_loop() -> None:
        while True:
            await asyncio.sleep(300)
            try:
                await store.sweep()
            except Exception:
                logger.exception("Session sweep error")

    sweep_task = asyncio.create_task(_sweep_loop())
    try:
        yield
    finally:
        sweep_task.cancel()
        try:
            await sweep_task
        except asyncio.CancelledError:
            pass


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = Settings()

    app = FastAPI(title="DERIVA Data Assistant", lifespan=_lifespan)
    app.state.settings = settings
    app.state.store = None  # set during lifespan

    @app.middleware("http")
    async def _anon_cookie_middleware(request: Request, call_next):  # type: ignore[misc]
        response = await call_next(request)
        new_anon = getattr(request.state, "new_anon_id", None)
        if new_anon is not None:
            anon_id, max_age = new_anon
            response.set_cookie(
                ANON_COOKIE_NAME,
                anon_id,
                httponly=True,
                samesite="lax",
                path="/",
                max_age=max_age,
            )
        return response

    app.include_router(auth_router)

    static_dir = pathlib.Path(__file__).parent / "static"
    if static_dir.exists() and any(static_dir.iterdir()):
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():  # type: ignore[misc]
        return JSONResponse({"status": "ok"})

    @app.get("/session-info")
    async def session_info(session: RequireSession, request: Request):  # type: ignore[misc]
        s: Settings = request.app.state.settings
        store = request.app.state.store
        cred = session.credenza_session or {}
        client_block = cred.get("client") or {}
        display_name = (
            "Anonymous"
            if session.bearer_token is None
            else (_extract_display_name(cred) or session.user_id)
        )
        full_name = cred.get("full_name") or client_block.get("full_name") or ""
        email = cred.get("email") or client_block.get("email") or ""
        is_anonymous = session.bearer_token is None
        rag_toggle_available = s.allow_rag_toggle and s.operating_tier in ("llm", "local") and not (s.rag_only_when_anonymous and is_anonymous)
        forced_rag_anon = s.rag_only_when_anonymous and is_anonymous
        rag_mode_active = s.operating_tier == "rag_only" or session.rag_only_override or forced_rag_anon
        lifetime_cost = await store.get_user_lifetime_cost(session.user_id) if not is_anonymous else None
        last_login = await store.get_user_last_seen(session.user_id) if not is_anonymous else None
        return JSONResponse(
            {
                "user_id": session.user_id,
                "session_id": session.session_id,
                "display_name": display_name,
                "full_name": full_name,
                "email": email,
                "catalog_mode": "default" if s.default_catalog_mode else "general",
                "operating_mode": s.operating_tier,
                "label": s.default_catalog_label or s.default_hostname or "",
                "hostname": s.default_hostname or "",
                "credenza_session": cred,
                "login_available": s.credenza_configured and is_anonymous,
                "rag_toggle_available": rag_toggle_available,
                "rag_mode_active": rag_mode_active,
                "rag_only_when_anonymous": s.rag_only_when_anonymous,
                "code_theme": s.code_theme,
                "show_response_cards": s.show_response_cards,
                "chat_align_left": s.chat_align_left,
                "turn_count": session.turn_count,
                "session_cost_usd": session.session_cost_usd if session.session_cost_usd else None,
                "lifetime_cost_usd": lifetime_cost if lifetime_cost else None,
                "last_login": last_login,
            }
        )

    @app.post("/rag-mode")
    async def set_rag_mode(body: _RagModeBody, session: RequireSession, request: Request):  # type: ignore[misc]
        """Set the per-session RAG-only override.

        Ignored when allow_rag_toggle is False or operating_tier is rag_only
        (those cases are handled by the frontend never showing the toggle).
        """
        s: Settings = request.app.state.settings
        store = request.app.state.store
        is_anonymous = session.bearer_token is None
        forced_rag_anon = s.rag_only_when_anonymous and is_anonymous
        if s.allow_rag_toggle and s.operating_tier in ("llm", "local") and not forced_rag_anon:
            session.rag_only_override = body.enabled
            await store.set(user_session_key(session.user_id), session)
        rag_mode_active = s.operating_tier == "rag_only" or session.rag_only_override or forced_rag_anon
        return JSONResponse({"rag_mode_active": rag_mode_active})

    @app.get("/history")
    async def get_history(session: RequireSession):  # type: ignore[misc]
        """Return display-friendly conversation history for the current user."""
        messages = []
        for msg in session.history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                if isinstance(content, str):
                    messages.append({"role": "user", "content": content})
            elif role == "assistant":
                # OpenAI format: content is a string, tool_calls is a separate list
                tool_calls = msg.get("tool_calls") or []
                if isinstance(content, str) and content:
                    messages.append({"role": "assistant", "content": content})
                if tool_calls:
                    tool_names = [
                        tc.get("function", {}).get("name", "")
                        for tc in tool_calls
                    ]
                    messages.append({"role": "tool_use", "tools": tool_names})
        return JSONResponse({"messages": messages, "input_history": session.input_history})

    @app.delete("/history")
    async def clear_history(session: RequireSession, request: Request):  # type: ignore[misc]
        """Clear conversation history for the current user."""
        store = request.app.state.store
        session.history = []
        session.input_history = []
        session.tools = None
        session.schema_primed = False
        session.primed_schema = ""
        session.primed_guides = ""
        session.primed_ermrest = ""
        await store.set(user_session_key(session.user_id), session)
        await store.delete(history_key(session.user_id))
        audit_event("history_cleared", user_id=session.user_id)
        return JSONResponse({"status": "ok"})

    @app.post("/chat")
    async def chat(body: ChatRequest, session: RequireSession, request: Request):  # type: ignore[misc]
        store = request.app.state.store
        s: Settings = request.app.state.settings

        if len(body.message) > s.max_message_length:
            return JSONResponse(
                {"detail": f"Message exceeds maximum length of {s.max_message_length} characters"},
                status_code=400,
            )

        # Detect stale sessions: if the client supplied a session_id that does
        # not match the resolved session, the server-side session has expired
        # and a new (anonymous) one was issued for this request.  Return a
        # dedicated error so the frontend can show a proper "session expired"
        # message rather than silently answering as an anonymous user.
        if body.session_id and body.session_id != session.session_id:
            return JSONResponse(
                {"error": "session_expired", "detail": "Your session has expired. Please log in again."},
                status_code=401,
            )

        # Update general-purpose catalog context if provided
        if body.hostname:
            session.gp_hostname = body.hostname
        if body.catalog_id:
            session.gp_catalog_id = body.catalog_id

        # Append to server-side input history (dedup consecutive; cap at 200).
        if not session.input_history or session.input_history[-1] != body.message:
            session.input_history.append(body.message)
            if len(session.input_history) > 200:
                session.input_history = session.input_history[-200:]

        session.turn_count += 1
        t0 = time.monotonic()
        cancelled = asyncio.Event()

        async def _event_stream():
            cost_before = session.session_cost_usd
            prompt_tokens_before = session.session_prompt_tokens
            completion_tokens_before = session.session_completion_tokens
            cache_read_before = session.session_cache_read_tokens
            cache_creation_before = session.session_cache_creation_tokens
            _turn_summary_emitted = False
            _turn_error: str | None = None
            try:
                async for event in run_chat_turn(body.message, session, s, cancelled=cancelled):
                    # Check if client disconnected between yields
                    if await request.is_disconnected():
                        cancelled.set()
                        logger.info("Client disconnected, cancelling chat for %s", session.user_id)
                        break
                    event_type = event.get("type")
                    if event_type == "text":
                        yield f"data: {json.dumps(event['content'])}\n\n"
                    elif event_type == "status":
                        yield f"event: status\ndata: {json.dumps(event)}\n\n"
                    elif event_type == "turn_summary":
                        # Consumed here; not forwarded to the SSE client.
                        _turn_summary_emitted = True
                        latency_ms = round((time.monotonic() - t0) * 1000)
                        turn_cost = session.session_cost_usd - cost_before
                        turn_prompt_tokens = session.session_prompt_tokens - prompt_tokens_before
                        turn_completion_tokens = session.session_completion_tokens - completion_tokens_before
                        turn_cache_read_tokens = session.session_cache_read_tokens - cache_read_before
                        turn_cache_creation_tokens = session.session_cache_creation_tokens - cache_creation_before
                        diag_fields: dict[str, object] = {
                            k: event[k] for k in (
                                "user_query", "response_text", "response_compressed",
                                "tool_inputs", "tool_outputs"
                            ) if k in event
                        }
                        if s.audit_diagnostic:
                            # Lifetime cost = durable total before this turn + this turn's delta.
                            # Avoids a DB round-trip while still giving the post-turn running total.
                            prior_lifetime = await store.get_user_lifetime_cost(session.user_id)
                            diag_fields["lifetime_cost_usd"] = round(
                                prior_lifetime + (turn_cost if turn_cost > 0 else 0.0), 6
                            )
                        audit_event(
                            "chat_turn",
                            session_id=session.session_id,
                            user_id=session.user_id,
                            turn=session.turn_count,
                            tools_invoked=event.get("tools_invoked", []),
                            rag_triggered=event.get("rag_triggered", False),
                            rag_document_count=event.get("rag_document_count", 0),
                            rag_documents=event.get("rag_documents", []),
                            rag_scores=event.get("rag_scores", []),
                            response_latency_ms=latency_ms,
                            cost_usd=round(turn_cost, 6) if turn_cost > 0 else 0.0,
                            prompt_tokens=turn_prompt_tokens,
                            completion_tokens=turn_completion_tokens,
                            cache_read_tokens=turn_cache_read_tokens,
                            cache_creation_tokens=turn_cache_creation_tokens,
                            model=event.get("model"),
                            **diag_fields,
                        )
                    else:
                        yield f"event: tool\ndata: {json.dumps(event)}\n\n"
            except ChatCancelled:
                logger.info("Chat cancelled for %s", session.user_id)
                audit_event("chat_cancelled",
                            session_id=session.session_id,
                            user_id=session.user_id,
                            turn=session.turn_count,
                            duration_ms=round((time.monotonic() - t0) * 1000))
            except MCPAuthError:
                _turn_error = "MCPAuthError"
                audit_event("chat_error", user_id=session.user_id, error_type="MCPAuthError")
                if session.bearer_token is None:
                    detail = "Login required -- please log in to use this feature"
                else:
                    detail = "Session expired -- please log in again"
                yield f"event: error\ndata: {json.dumps({'error': 'auth', 'detail': detail})}\n\n"
            except BaseExceptionGroup as exc:
                # streamablehttp_client wraps TaskGroup errors; a 401 from MCP
                # surfaces as an ExceptionGroup rather than a bare MCPAuthError
                # when _connect's try/except is bypassed by the group boundary.
                http_errs = [e for e in exc.exceptions if isinstance(e, httpx.HTTPStatusError)]
                if http_errs and http_errs[0].response.status_code == 401:
                    _turn_error = "MCPAuthError"
                    audit_event("chat_error", user_id=session.user_id, error_type="MCPAuthError")
                    detail = ("Login required -- please log in to use this feature"
                              if session.bearer_token is None
                              else "Session expired -- please log in again")
                    yield f"event: error\ndata: {json.dumps({'error': 'auth', 'detail': detail})}\n\n"
                else:
                    _turn_error = f"ExceptionGroup: {exc}"
                    audit_event("chat_error", user_id=session.user_id,
                                error_type="ExceptionGroup", detail=str(exc))
                    logger.exception("Chat ExceptionGroup for user %s", session.user_id)
                    yield f"event: error\ndata: {json.dumps({'error': 'internal', 'detail': str(exc)})}\n\n"
            except Exception as exc:
                _turn_error = f"{type(exc).__name__}: {exc}"
                audit_event(
                    "chat_error",
                    user_id=session.user_id,
                    error_type=type(exc).__name__,
                    detail=str(exc),
                )
                logger.exception("Chat error for user %s", session.user_id)
                yield f"event: error\ndata: {json.dumps({'error': 'internal', 'detail': str(exc)})}\n\n"
            finally:
                # If the turn was cut short by an error before turn_summary was
                # yielded, emit a minimal chat_turn record so every request has one.
                if not _turn_summary_emitted and _turn_error is not None:
                    audit_event(
                        "chat_turn",
                        session_id=session.session_id,
                        user_id=session.user_id,
                        tools_invoked=[],
                        rag_triggered=False,
                        rag_document_count=0,
                        rag_documents=[],
                        rag_scores=[],
                        response_latency_ms=round((time.monotonic() - t0) * 1000),
                        cost_usd=0.0,
                        model=s.llm_model,
                        error=_turn_error,
                    )
                # Persist updated session (history + cached tools) keyed by stable user_id
                await store.set(user_session_key(session.user_id), session)
                # Durably accumulate per-user lifetime LLM spend and token counts.
                turn_cost = session.session_cost_usd - cost_before
                if turn_cost > 0:
                    await store.increment_user_cost(
                        session.user_id,
                        turn_cost,
                        prompt_tokens=session.session_prompt_tokens - prompt_tokens_before,
                        completion_tokens=session.session_completion_tokens - completion_tokens_before,
                        cache_read_tokens=session.session_cache_read_tokens - cache_read_before,
                        cache_creation_tokens=session.session_cache_creation_tokens - cache_creation_before,
                    )
                # Keep the user identity table current for authenticated users.
                if session.bearer_token is not None:
                    cred = session.credenza_session or {}
                    client_block = cred.get("client") or {}
                    _email = cred.get("email") or client_block.get("email") or ""
                    _full_name = cred.get("full_name") or client_block.get("full_name") or ""
                    await store.upsert_user_identity(session.user_id, _email, _full_name)
                # Persist full (untrimmed) history with longer TTL so it survives session expiry
                full = getattr(session, "full_history", session.history)
                hist_session = Session(user_id=session.user_id, history=full, input_history=session.input_history)
                await store.set(history_key(session.user_id), hist_session, ttl=s.history_ttl)
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    @app.get("/")
    async def index(request: Request):  # type: ignore[misc]
        template = static_dir / "index.html"
        if not template.exists():
            return JSONResponse({"status": "ok", "phase": "3 -- UI not yet built"})
        s: Settings = request.app.state.settings
        page = template.read_text(encoding="utf-8")
        page = page.replace("{{HEADER_TITLE}}", html.escape(s.header_title))
        logo_url = html.escape(s.header_logo_url) if s.header_logo_url else ""
        page = page.replace("{{HEADER_LOGO_URL}}", logo_url)
        page = page.replace("{{HEADER_BG_COLOR}}", html.escape(s.header_bg_color))
        page = page.replace("{{INPUT_AREA_BG_COLOR}}", html.escape(s.input_area_bg_color))
        page = page.replace("{{CHAT_BG_COLOR}}", html.escape(s.chat_bg_color))
        # Inject <base href="..."> so relative asset paths resolve correctly
        # whether the browser URL has a trailing slash or not.
        if s.public_url:
            base_path = urlparse(s.public_url).path.rstrip("/") + "/"
            page = page.replace("<head>", f'<head><base href="{html.escape(base_path)}">', 1)
        return HTMLResponse(page)

    return app


def main() -> None:  # pragma: no cover
    """CLI entry point."""
    import uvicorn

    settings = Settings()
    settings.validate_for_http()

    _init_logging(debug=settings.debug, app_use_syslog=settings.app_use_syslog)
    _init_access_logging(settings)

    init_audit_logger(use_syslog=settings.audit_use_syslog)
    app = create_app(settings)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="debug" if settings.debug else "info",
        log_config=None,
    )
