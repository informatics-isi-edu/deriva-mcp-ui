"""FastAPI application: route registration and lifespan."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import pathlib
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .audit import audit_event, init_audit_logger
from .auth import ANON_COOKIE_NAME, RequireSession, _extract_display_name, user_session_key
from .auth import router as auth_router
from .chat import run_chat_turn
from .config import Settings
from .mcp_client import MCPAuthError
from .storage import create_store

logger = logging.getLogger(__name__)


def _init_logging(debug: bool = False) -> None:  # pragma: no cover
    """Configure the root deriva_mcp_ui logger.

    Always adds a stderr stream handler so logs appear in docker logs.
    Also adds a syslog handler when /dev/log is available, for production
    deployments that forward syslog to a central collector.
    """
    import os
    from logging.handlers import SysLogHandler

    fmt_stream = logging.Formatter(
        "%(asctime)s [%(process)d] [%(levelname)s] [%(name)s] - %(message)s"
    )
    fmt_syslog = logging.Formatter(
        "[%(process)d] [%(levelname)s] [%(name)s] - %(message)s"
    )

    root = logging.getLogger("deriva_mcp_ui")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt_stream)
    root.addHandler(stream_handler)

    syslog_socket = "/dev/log"
    if os.path.exists(syslog_socket) and os.access(syslog_socket, os.W_OK):
        try:
            sh = SysLogHandler(address=syslog_socket, facility=SysLogHandler.LOG_LOCAL1)
            sh.ident = "deriva-mcp-ui: "
            sh.setFormatter(fmt_syslog)
            root.addHandler(sh)
        except Exception:
            pass

    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.propagate = False

    # Suppress per-request noise from httpx/httpcore
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


class ChatRequest(BaseModel):
    message: str
    hostname: str = ""
    catalog_id: str = ""


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

    app = FastAPI(title="DERIVA Chatbot", lifespan=_lifespan)
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
        display_name = (
            "Anonymous"
            if not s.auth_enabled
            else (_extract_display_name(session.credenza_session) or session.user_id)
        )
        return JSONResponse(
            {
                "user_id": session.user_id,
                "display_name": display_name,
                "catalog_mode": "default" if s.default_catalog_mode else "general",
                "label": s.default_catalog_label or s.default_hostname or "",
                "credenza_session": session.credenza_session,
            }
        )

    @app.post("/chat")
    async def chat(body: ChatRequest, session: RequireSession, request: Request):  # type: ignore[misc]
        store = request.app.state.store
        s: Settings = request.app.state.settings

        if len(body.message) > s.max_message_length:
            return JSONResponse(
                {"detail": f"Message exceeds maximum length of {s.max_message_length} characters"},
                status_code=400,
            )

        # Update general-purpose catalog context if provided
        if body.hostname:
            session.gp_hostname = body.hostname
        if body.catalog_id:
            session.gp_catalog_id = body.catalog_id

        audit_event("chat_request", user_id=session.user_id, msg_len=len(body.message))
        t0 = time.monotonic()

        async def _event_stream():
            try:
                async for event in run_chat_turn(body.message, session, s):
                    if event.get("type") == "text":
                        yield f"data: {json.dumps(event['content'])}\n\n"
                    else:
                        yield f"event: tool\ndata: {json.dumps(event)}\n\n"
                audit_event(
                    "chat_complete",
                    user_id=session.user_id,
                    duration_ms=round((time.monotonic() - t0) * 1000),
                )
            except MCPAuthError:
                audit_event("chat_error", user_id=session.user_id, error_type="MCPAuthError")
                yield f"event: error\ndata: {json.dumps({'error': 'auth', 'detail': 'Session expired -- please log in again'})}\n\n"
            except Exception as exc:
                audit_event(
                    "chat_error",
                    user_id=session.user_id,
                    error_type=type(exc).__name__,
                    detail=str(exc),
                )
                logger.exception("Chat error for user %s", session.user_id)
                yield f"event: error\ndata: {json.dumps({'error': 'internal', 'detail': str(exc)})}\n\n"
            finally:
                # Persist updated session (history + cached tools) keyed by stable user_id
                await store.set(user_session_key(session.user_id), session)
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
        return HTMLResponse(page)

    return app


def main() -> None:  # pragma: no cover
    """CLI entry point."""
    import uvicorn

    settings = Settings()
    settings.validate_for_http()

    _init_logging(debug=settings.debug)

    init_audit_logger(use_syslog=True)
    app = create_app(settings)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="debug" if settings.debug else "info",
    )
