"""Credenza OAuth 2.0 client routes: /login, /callback, /logout.

Uses the authorization_code flow with PKCE (RFC 7636).

Session storage uses a two-key pattern so conversation history survives
re-authentication (e.g. when the bearer token expires and the user logs in again):

  tok:{bearer_token}  ->  minimal Session entry (user_id only)   TTL = session_ttl
  uid:{user_id}       ->  full Session with history               sliding TTL

The cookie value IS the opaque bearer token (no separate signing key required).
On re-auth, the uid: entry is updated with the new token while retaining all
existing history, cached tools, and schema_primed state.  The expired tok: entry
is swept by the normal TTL mechanism.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from base64 import urlsafe_b64encode
from typing import Annotated, Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from .audit import audit_event
from .config import Settings
from .storage.base import Session

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


def _generate_code_verifier() -> str:
    """Return a 32-byte URL-safe base64-encoded code verifier (no padding)."""
    return urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()


def _code_challenge(verifier: str) -> str:
    """Return the S256 code challenge for a verifier."""
    digest = hashlib.sha256(verifier.encode()).digest()
    return urlsafe_b64encode(digest).rstrip(b"=").decode()


# ---------------------------------------------------------------------------
# Session store key helpers
# ---------------------------------------------------------------------------

COOKIE_NAME = "deriva_chatbot_session"
PKCE_COOKIE_NAME = "deriva_chatbot_pkce"
ANON_COOKIE_NAME = "deriva_chatbot_anon"

# Resource URI required by Credenza's GET /session endpoint.
# Requested alongside the MCP resource so the issued token satisfies both
# the MCP server's audience check and the /session identity fetch.
_DERIVA_SERVICE_RESOURCE = "urn:deriva:rest:service:all"

_TOKEN_PREFIX = "tok:"
_USER_PREFIX = "uid:"
_HISTORY_PREFIX = "history:"


def _token_key(bearer_token: str) -> str:
    """Store key for the token-to-user-id index entry."""
    return _TOKEN_PREFIX + bearer_token


def user_session_key(user_id: str) -> str:
    """Store key for the full persistent session (history, tools, etc.)."""
    return _USER_PREFIX + user_id


def history_key(user_id: str) -> str:
    """Store key for long-lived conversation history."""
    return _HISTORY_PREFIX + user_id


# ---------------------------------------------------------------------------
# Cookie helpers
# ---------------------------------------------------------------------------


def _get_session_id(request: Request, settings: Settings) -> str | None:  # noqa: ARG001
    """Return the bearer token from the session cookie, or None if absent."""
    return request.cookies.get(COOKIE_NAME) or None


def _set_session_cookie(response: Response, bearer_token: str, settings: Settings) -> None:
    response.set_cookie(
        COOKIE_NAME,
        bearer_token,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
        max_age=settings.session_ttl,
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, secure=True, samesite="lax")


# ---------------------------------------------------------------------------
# Require-session dependency
# ---------------------------------------------------------------------------


async def require_session(request: Request, response: Response) -> Session:
    """FastAPI dependency: return the current Session or raise 401.

    In anonymous mode (DERIVA_CHATBOT_ANONYMOUS_MODE=true), creates or retrieves
    a per-browser anonymous session keyed by a cookie -- no login required.

    In normal mode, two store lookups:
      1. tok:{bearer} -> token index entry (yields user_id)
      2. uid:{user_id} -> full Session with history
    """
    settings: Settings = request.app.state.settings
    store = request.app.state.store

    # When Credenza is configured, always try the bearer token first so that
    # an authenticated user in allow_anonymous mode gets their real session.
    if settings.credenza_configured:
        bearer_token = _get_session_id(request, settings)
        if bearer_token is not None:
            tok_entry = await store.get(_token_key(bearer_token))
            if tok_entry is not None:
                session = await store.get(user_session_key(tok_entry.user_id))
                if session is not None:
                    session.last_active = time.time()
                    await store.set(user_session_key(session.user_id), session)
                    return session
        # No valid bearer token -- enforce login or fall through to anonymous.
        if settings.auth_enabled:
            raise HTTPException(status_code=401, detail="Not authenticated")

    return await _get_or_create_anonymous_session(request, response, settings, store)


async def _get_or_create_anonymous_session(
    request: Request,
    response: Response,  # noqa: ARG001
    settings: Settings,
    store: Any,
) -> Session:
    """Get the existing anonymous session from the cookie, or create a new one.

    Each browser gets its own session (isolated history) via an opaque random ID
    stored in a cookie.  The session is persisted in the store under
    uid:anonymous/<anon_id> so that the chat endpoint's normal store.set() call
    preserves history across turns.

    When a new session is created the anon_id is stashed on request.state so
    that the _anon_cookie_middleware in server.py can attach the Set-Cookie
    header to the outgoing response (FastAPI does not merge cookies set on the
    injected Response object when the endpoint returns a JSONResponse directly).
    """
    anon_id = request.cookies.get(ANON_COOKIE_NAME)
    if anon_id:
        session = await store.get(user_session_key(f"anonymous/{anon_id}"))
        if session is not None:
            session.last_active = time.time()
            await store.set(user_session_key(session.user_id), session)
            return session
        # Session expired but cookie still valid -- restore history if available
        user_id = f"anonymous/{anon_id}"
        hist_entry = await store.get(history_key(user_id))
    else:
        anon_id = urlsafe_b64encode(os.urandom(16)).rstrip(b"=").decode()
        hist_entry = None

    # New anonymous session (possibly with restored history)
    user_id = f"anonymous/{anon_id}"
    now = time.time()
    session = Session(
        user_id=user_id,
        bearer_token=None,
        created_at=now,
        last_active=now,
    )
    if hist_entry is not None:
        session.history = hist_entry.history
        session.input_history = hist_entry.input_history
    await store.set(user_session_key(session.user_id), session)
    # Stash cookie info for _anon_cookie_middleware to attach to the response
    request.state.new_anon_id = (anon_id, settings.session_ttl)
    audit_event("anonymous_session_created")
    return session


RequireSession = Annotated[Session, Depends(require_session)]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/login")
async def login(request: Request) -> Response:
    """Start the OAuth authorization_code + PKCE flow.

    In anonymous mode, redirect immediately to / -- no login is needed.
    """
    settings: Settings = request.app.state.settings

    if not settings.credenza_configured:
        return RedirectResponse("/", status_code=302)

    verifier = _generate_code_verifier()
    challenge = _code_challenge(verifier)
    state = urlsafe_b64encode(os.urandom(16)).rstrip(b"=").decode()

    # Two resource params: MCP resource (for MCP server audience check) and
    # the DERIVA service resource (required by Credenza GET /session for identity fetch).
    params = [
        ("response_type", "code"),
        ("client_id", settings.client_id),
        ("redirect_uri", f"{settings.public_url}/callback"),
        ("scope", "openid email profile"),
        ("resource", settings.mcp_resource),
        ("resource", _DERIVA_SERVICE_RESOURCE),
        ("state", state),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
    ]
    authorize_url = f"{settings.credenza_url}/authorize?{urlencode(params)}"
    audit_event("login_redirect", client_id=settings.client_id)

    pkce_payload = json.dumps({"verifier": verifier, "state": state})
    response = RedirectResponse(authorize_url, status_code=302)
    response.set_cookie(
        PKCE_COOKIE_NAME,
        pkce_payload,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=300,  # 5 minutes -- matches Credenza pending_consent TTL
    )
    return response


@router.get("/callback")
async def callback(request: Request, code: str = "", state: str = "", error: str = "") -> Response:
    """Handle the Credenza redirect after user authentication."""
    settings: Settings = request.app.state.settings
    store = request.app.state.store

    if error:
        audit_event("login_failed", reason="oauth_error", error=error)
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")

    pkce_raw = request.cookies.get(PKCE_COOKIE_NAME)
    if not pkce_raw:
        raise HTTPException(status_code=400, detail="Missing PKCE cookie")

    try:
        pkce = json.loads(pkce_raw)
    except (ValueError, KeyError):
        raise HTTPException(status_code=400, detail="Invalid PKCE cookie")

    if pkce.get("state") != state:
        audit_event("login_failed", reason="state_mismatch")
        raise HTTPException(status_code=400, detail="State mismatch -- possible CSRF")

    # Exchange code for bearer token
    token_url = f"{settings.remap_url(settings.credenza_url)}/token"
    async with httpx.AsyncClient(verify=settings.ssl_verify) as client:
        resp = await client.post(
            token_url,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": f"{settings.public_url}/callback",
                "client_id": settings.client_id,
                "code_verifier": pkce["verifier"],
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if resp.status_code != 200:
        audit_event("login_failed", reason="token_exchange_failed", status=resp.status_code)
        raise HTTPException(status_code=502, detail=f"Token exchange failed: {resp.status_code}")

    token_data = resp.json()
    bearer_token = token_data.get("access_token") or token_data.get("token")
    if not bearer_token:
        raise HTTPException(status_code=502, detail="No access_token in token response")

    credenza_data = await _fetch_credenza_session(bearer_token, settings)
    user_id = _extract_user_id(credenza_data)

    # Preserve existing session history on re-authentication
    existing = await store.get(user_session_key(user_id))
    now = time.time()
    if existing is not None:
        existing.bearer_token = bearer_token
        existing.credenza_session = credenza_data
        existing.last_active = now
        existing.tools = None  # invalidate cached tool list -- new token may have different scope
        session = existing
    else:
        session = Session(
            user_id=user_id,
            bearer_token=bearer_token,
            credenza_session=credenza_data,
            created_at=now,
            last_active=now,
        )
        # Restore conversation history from long-lived history store
        hist_entry = await store.get(history_key(user_id))
        if hist_entry is not None:
            session.history = hist_entry.history
            session.input_history = hist_entry.input_history

    # Store full session keyed by user_id (persistent across re-auth)
    await store.set(user_session_key(user_id), session)
    # Store token index entry for fast dereference (expires with the token)
    tok_entry = Session(user_id=user_id, bearer_token=bearer_token, created_at=now, last_active=now)
    await store.set(_token_key(bearer_token), tok_entry)

    audit_event("login_success", user_id=user_id)
    response = RedirectResponse(f"{settings.public_url}/", status_code=302)
    _set_session_cookie(response, bearer_token, settings)
    response.delete_cookie(PKCE_COOKIE_NAME)
    return response


@router.get("/logout")
async def logout(request: Request) -> Response:
    """Clear the local session and redirect to Credenza /logout.

    In anonymous mode, clears the anonymous session cookie and returns to /.
    In normal mode, clears the local session store entry and redirects the
    browser to Credenza's /logout endpoint, which revokes the token, clears
    the Credenza session cookie, and triggers the IDP logout redirect.
    """
    settings: Settings = request.app.state.settings
    store = request.app.state.store

    bearer_token = _get_session_id(request, settings)
    is_anonymous = bearer_token is None or not settings.credenza_configured
    if is_anonymous:
        anon_id = request.cookies.get(ANON_COOKIE_NAME)
        if anon_id:
            session = await store.get(user_session_key(f"anonymous/{anon_id}"))
            if session is not None:
                audit_event("logout", user_id=session.user_id)
                await store.delete(user_session_key(session.user_id))
        response = RedirectResponse(f"{settings.public_url}/", status_code=302)
        response.delete_cookie(ANON_COOKIE_NAME, path=f"{settings.public_url}/", httponly=True, samesite="lax")
        return response

    if bearer_token:
        tok_entry = await store.get(_token_key(bearer_token))
        if tok_entry:
            audit_event("logout", user_id=tok_entry.user_id)
            await store.delete(user_session_key(tok_entry.user_id))
        await store.delete(_token_key(bearer_token))

    # Call Credenza /logout server-side with the bearer token so Credenza can
    # identify the session, revoke the token, and return its IDP logout redirect.
    # follow_redirects=False: we want the redirect URL, not the IDP page itself.
    # Redirect the browser to that URL so the IDP end-session flow completes.
    redirect_target = f"{settings.public_url}/"
    if bearer_token:
        try:
            async with httpx.AsyncClient(verify=settings.ssl_verify) as client:
                resp = await client.get(
                    f"{settings.remap_url(settings.credenza_url)}/logout",
                    headers={"Authorization": f"Bearer {bearer_token}"},
                    follow_redirects=False,
                    timeout=5,
                )
            if resp.status_code in (301, 302, 303, 307, 308):
                # Standard mode: Credenza returns a redirect directly to the IDP
                location = resp.headers.get("location")
                if location:
                    redirect_target = location
                    logger.debug("Credenza logout redirect: %s", location)
            elif resp.status_code == 200:
                # Legacy API mode: Credenza returns {"logout_url": "..."}
                try:
                    logout_url = resp.json().get("logout_url")
                    if logout_url:
                        redirect_target = logout_url
                        logger.debug("Credenza logout_url (legacy): %s", logout_url)
                except Exception:
                    pass
            else:
                logger.warning("Credenza /logout returned unexpected status %d", resp.status_code)
        except Exception as exc:
            logger.warning("Credenza logout call failed (non-fatal): %s", exc)

    response = RedirectResponse(redirect_target, status_code=302)
    _clear_session_cookie(response)
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_user_id(credenza_data: dict) -> str:
    """Return the iss/sub composite user_id from a Credenza session dict.

    Non-legacy: top-level "id" field.
    Legacy: client.id field.
    Falls back to "unknown".
    """
    client_block = credenza_data.get("client") or {}
    return credenza_data.get("id") or client_block.get("id") or "unknown"


def _extract_display_name(credenza_data: dict) -> str:
    """Return a human-readable display name from a Credenza session dict.

    Non-legacy: full_name > preferred_username > email.
    Legacy: client.full_name > client.display_name > client.email.
    Falls back to "".
    """
    client_block = credenza_data.get("client") or {}
    return (
        credenza_data.get("full_name")
        or credenza_data.get("preferred_username")
        or credenza_data.get("email")
        or client_block.get("full_name")
        or client_block.get("display_name")
        or client_block.get("email")
        or ""
    )


async def _fetch_credenza_session(bearer_token: str, settings: Settings) -> dict:
    """Fetch the full Credenza GET /session response dict for a bearer token.

    Returns an empty dict on failure so callers can handle gracefully.

    TODO: replace with GET /userinfo once that endpoint is implemented in Credenza
    (see credenza workplan Phase 9).
    """
    session_url = f"{settings.remap_url(settings.credenza_url)}/session"
    async with httpx.AsyncClient(verify=settings.ssl_verify) as client:
        resp = await client.get(session_url, headers={"Authorization": f"Bearer {bearer_token}"})
    if resp.status_code != 200:
        return {}
    return resp.json()
