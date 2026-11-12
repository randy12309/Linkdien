"""FastMCP tool service. Run standalone with: python -m mcp_server.server

Identity is never a model-supplied argument: account tools take a short-lived
`principal` token minted by the backend and verified here.

Member data comes from the BMOWNER tables: USER_ACCOUNT (login), EPAYMENT_METHOD (saved methods) and
EPAYMENT (payments). An outstanding premium is an EPAYMENT row still staged (EPMT_STATUS 'P'); a paid one
is synced ('S') and approved ('A'). There is no currency or due-date column in the legacy schema.
"""
import json
import logging
import math
import os
import secrets
import time
from pathlib import Path
from typing import Any

import jwt
from dotenv import load_dotenv
from fastmcp import FastMCP

from mcp_server import cache, db, vectordb

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ACCOUNTS_FILE = Path(__file__).parent.parent / "app" / "data" / "accounts.json"
MCP_SECRET = os.getenv("MCP_INTERNAL_SECRET", "dev-only-mcp-secret-change-me-0123456789")
PRINCIPAL_TTL_SECONDS = 60
QUOTE_TTL_SECONDS = 300

mcp = FastMCP("plan-assistant-tools")

# Simulated payment state for the JSON demo mode (MySQL off). The participant tables are authoritative.
_paid: set[str] = set()
_quotes: dict[str, dict[str, Any]] = {}


def _plan_settings() -> dict[str, Any] | None:
    """Plan numbers from the database (Qdrant); None if unavailable or incomplete."""
    try:
        raw = vectordb.get_settings()
        return {
            "eligibility_threshold": float(raw["eligibility_threshold"]),
            "quarterly_premium": float(raw["quarterly_premium"]),
            "base_coverage_ratio": float(raw["base_coverage_ratio"]),
            "currency": raw["currency"],
        }
    except Exception:
        logging.getLogger(__name__).warning("Plan settings unavailable", exc_info=True)
        return None


def mint_principal(username: str) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": username, "aud": "mcp", "iat": now, "exp": now + PRINCIPAL_TTL_SECONDS},
        MCP_SECRET,
        algorithm="HS256",
    )


def _ok(data: dict, source: str) -> dict:
    return {"status": "ok", "source": source, "data": data}


def _err(error_type: str, message: str) -> dict:
    return {"status": "error", "error_type": error_type, "message": message}


def _authenticate(principal: str) -> str | None:
    try:
        claims = jwt.decode(principal, MCP_SECRET, algorithms=["HS256"], audience="mcp")
    except jwt.PyJWTError:
        return None
    return claims.get("sub")


def _premium(username: str) -> dict | None:
    """The member's premium standing: the oldest staged EPAYMENT row, else their last payment.

    None means no such member. Falls back to the accounts.json demo record when MySQL is off.
    """
    if db.enabled():
        return db.get_premium(username)
    accounts = json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
    account = accounts.get(username)
    if account is None:
        return None
    premium = dict(account["premium"])
    if username in _paid:
        premium["status"] = "Paid"
    return premium


@mcp.tool
def get_balance(principal: str) -> dict:
    """Return the caller's current premium balance and status."""
    user = _authenticate(principal)
    if user is None:
        return _err("unauthorized", "Invalid or expired principal.")
    premium = cache.get_or_load(user, "balance", lambda: _premium(user))
    if premium is None:
        return _err("not_found", "No account found.")
    return _ok(premium, "epayment")


@mcp.tool
def get_policy_info(topic: str) -> dict:
    """Return public policy text relevant to the question (top-5 similarity search). No authentication required."""
    try:
        chunks = vectordb.search_policy(topic)
    except Exception:
        return _err("unavailable", "Policy store is unavailable.")
    if not chunks:
        return _err("not_found", "No policy information for that topic.")
    return _ok({"topic": "uploaded policy", "text": "\n\n".join(chunks)}, "policy:uploaded")


@mcp.tool
def explain_premium(principal: str) -> dict:
    """Return the caller's premium with its component breakdown."""
    user = _authenticate(principal)
    if user is None:
        return _err("unauthorized", "Invalid or expired principal.")
    premium = _premium(user)
    if premium is None:
        return _err("not_found", "No account found.")
    plan = _plan_settings()
    if plan is None:
        return _err("unavailable", "Plan settings are unavailable.")
    amount = premium["amount"]
    base = round(amount * plan["base_coverage_ratio"], 2)
    breakdown = {"base_coverage": base, "taxes_and_fees": round(amount - base, 2)}
    return _ok({**premium, "breakdown": breakdown}, "epayment")


@mcp.tool
def get_eligibility(principal: str) -> dict:
    """Return whether the caller's total paid premiums reach the insurance and pension threshold."""
    user = _authenticate(principal)
    if user is None:
        return _err("unauthorized", "Invalid or expired principal.")
    return cache.get_or_load(user, "eligibility", lambda: _eligibility(user))


def _eligibility(user: str) -> dict:
    plan = _plan_settings()
    if plan is None:
        return _err("unavailable", "Plan settings are unavailable.")
    threshold, quarterly = plan["eligibility_threshold"], plan["quarterly_premium"]
    if db.enabled():
        total_paid = db.total_paid(user)
        if total_paid is None:
            return _err("not_found", "No account found.")
    else:
        premium = _premium(user)
        if premium is None:
            return _err("not_found", "No account found.")
        total_paid = premium["amount"] if user in _paid else 0.0
    remaining = max(threshold - total_paid, 0.0)
    return _ok(
        {
            "eligible": total_paid >= threshold,
            "total_paid": total_paid,
            "threshold": threshold,
            "remaining": remaining,
            "quarters_remaining": math.ceil(remaining / quarterly),
            "quarterly_premium": quarterly,
            "currency": plan["currency"],
        },
        "eligibility",
    )


@mcp.tool
def get_user_account(principal: str) -> dict:
    """Return the caller's USER_ACCOUNT record (status, contact, verification flags, te_suid). No password data."""
    user = _authenticate(principal)
    if user is None:
        return _err("unauthorized", "Invalid or expired principal.")
    if not db.enabled():
        return _err("unavailable", "Database is not enabled.")
    row = db.get_user_account(user)
    if row is None:
        return _err("not_found", "No account found.")
    return _ok(row, "user_account")


@mcp.tool
def list_payment_methods(principal: str) -> dict:
    """Return the caller's saved payment methods (EPAYMENT_METHOD): mode, status, card type, nickname. No tokens."""
    user = _authenticate(principal)
    if user is None:
        return _err("unauthorized", "Invalid or expired principal.")
    if not db.enabled():
        return _err("unavailable", "Database is not enabled.")
    # Use Valkey cache for payment methods (5 min TTL)
    cached = cache.get_account_info(user, "payment_methods")
    if cached:
        return _ok(cached, "epayment_method:cache")
    items = db.list_payment_methods(user)
    if items is None:
        return _err("not_found", "No account found.")
    result = {"items": items}
    cache.set_account_info(user, "payment_methods", result, ttl=300)
    return _ok(result, "epayment_method")


@mcp.tool
def list_epayments(principal: str, status: str = "", limit: int = 20) -> dict:
    """Return the caller's payments (EPAYMENT), newest first. status: P pending, S synced, F failed; empty for all."""
    user = _authenticate(principal)
    if user is None:
        return _err("unauthorized", "Invalid or expired principal.")
    if status not in ("", "P", "S", "F"):
        return _err("invalid_argument", "status must be P, S, F or empty.")
    if not db.enabled():
        return _err("unavailable", "Database is not enabled.")
    # Use Valkey cache for payment history (2 min TTL)
    cache_key = f"payments:{status}:{limit}"
    cached = cache.get_account_info(user, cache_key)
    if cached:
        return _ok(cached, "epayment:cache")
    items = db.list_epayments(user, status or None, limit)
    if items is None:
        return _err("not_found", "No account found.")
    result = {"items": items}
    cache.set_account_info(user, cache_key, result, ttl=120)
    return _ok(result, "epayment")


@mcp.tool
def get_due_payments(principal: str) -> dict:
    """Return the caller's outstanding (unpaid) premiums: EPAYMENT rows still staged, oldest first. Read-only."""
    user = _authenticate(principal)
    if user is None:
        return _err("unauthorized", "Invalid or expired principal.")
    if db.enabled():
        return _ok({"items": db.list_due(user)}, "epayment")
    premium = _premium(user)
    if premium is None:
        return _err("not_found", "No account found.")
    items = [] if premium["status"] == "Paid" else [{**premium, "payment_id": f"demo-{user}"}]
    return _ok({"items": items}, "epayment")


@mcp.tool
def quote_payment(principal: str) -> dict:
    """Create a single-use quote to pay the caller's outstanding premium. Moves no money."""
    user = _authenticate(principal)
    if user is None:
        return _err("unauthorized", "Invalid or expired principal.")
    premium = _premium(user)
    if premium is None:
        return _err("not_found", "No account found.")
    if premium["status"] != "Pending":
        return _err("nothing_due", "No premium is currently due.")
    quote_id = secrets.token_urlsafe(12)
    _quotes[quote_id] = {
        "user": user, "amount": premium["amount"], "payment_id": premium.get("payment_id"),
        "expires": time.time() + QUOTE_TTL_SECONDS,
    }
    return _ok(
        {"quote_id": quote_id, "amount": premium["amount"], "currency": premium["currency"]},
        "payment-quote",
    )


@mcp.tool
def execute_payment(principal: str, quote_id: str) -> dict:
    """Execute a simulated payment for a quote owned by the caller. No live processor."""
    user = _authenticate(principal)
    if user is None:
        return _err("unauthorized", "Invalid or expired principal.")
    quote = _quotes.pop(quote_id, None)
    if quote is None or quote["user"] != user:
        return _err("invalid_quote", "Quote is unknown or already used.")
    if quote["expires"] < time.time():
        return _err("expired_quote", "Quote has expired.")
    if db.enabled():
        # Confirms the staged EPAYMENT row as synced and approved; a stale quote pays nothing.
        reference = db.mark_paid_id(user, quote["payment_id"], db.payment_reference())
        if reference is None:
            return _err("nothing_due", "No premium is currently due.")
        cache.invalidate_account(user)
        cache.invalidate_account_info(user, "payment_methods")
        cache.invalidate_account_info(user, "payments:")
        return _ok(
            {"amount": quote["amount"], "status": "Paid", "simulated": True, "reference": reference},
            "payment-simulation",
        )
    _paid.add(user)
    cache.invalidate_account(user)
    cache.invalidate_account_info(user, "payment_methods")
    cache.invalidate_account_info(user, "payments:")
    return _ok({"amount": quote["amount"], "status": "Paid", "simulated": True}, "payment-simulation")


def reset_state() -> None:
    _paid.clear()
    _quotes.clear()


if __name__ == "__main__":
    if vectordb.enabled():
        try:
            vectordb.ensure_ready()
        except Exception as exc:
            print(f"Warning: could not prepare the Qdrant collections: {exc}")
    mcp.run(transport="http", host="127.0.0.1", port=int(os.getenv("MCP_PORT", "8765")))
