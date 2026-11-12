"""Read-only administrator console API: database tables, chat history, embeddings and the agent/tool-call flow.

Every route requires an administrator and is written to the application log with the admin's username. Password hashes,
payment-card data and vectors' raw bytes are never returned. Tables are whitelisted, so no request can name a table or
column, and all values are passed as query parameters.
"""
import asyncio
import datetime
import decimal
import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.agents import flow_store
from app.agents.mcp_client import ALLOWED_TOOLS
from app.auth import admin_user
from app.users import hash_password
from mcp_server import db, vectordb

logger = logging.getLogger(__name__)


def _audit(request: Request, admin: str = Depends(admin_user)) -> str:
    logger.info("admin %s %s %s", admin, request.method, request.url.path)
    return admin


router = APIRouter(prefix="/api/v1/admin", tags=["admin"], dependencies=[Depends(_audit)])

MAX_FILTER_CHARS = 100
HISTORY_MAX_MESSAGES = 500
PREVIEW_CHARS = 300


def _col(key: str, label: str, kind: str = "text") -> dict:
    return {"key": key, "label": label, "kind": kind}


_TABLES: dict[str, dict] = {
    "admins": {
        "select": "ua.user_name AS username, p.display_name, COALESCE(p.email, ua.email_id) AS email, "
                  "a.granted_by, a.created_at",
        "from": "admins a JOIN user_account ua ON ua.account_id = a.account_id "
                "LEFT JOIN users_profile p ON p.te_suid = ua.te_suid",
        "order": "a.created_at, ua.user_name",
        "search": ["ua.user_name", "p.display_name", "p.email", "ua.email_id"],
        "filters": {"username": "ua.user_name"},
        "columns": [_col("username", "Username"), _col("display_name", "Name"), _col("email", "Email"),
                    _col("granted_by", "Granted by"), _col("created_at", "Added", "datetime")],
    },
    "users_profile": {
        "select": "ua.account_id, ua.user_name, ua.te_suid, p.display_name, "
                  "COALESCE(p.email, ua.email_id) AS email, ua.status_code, ua.created_date, p.updated_at",
        "from": "user_account ua LEFT JOIN users_profile p ON p.te_suid = ua.te_suid",
        "order": "ua.account_id",
        "search": ["ua.user_name", "ua.te_suid", "p.display_name", "p.email"],
        "filters": {"username": "ua.user_name", "status_code": "ua.status_code"},
        "columns": [_col("user_name", "Username"), _col("display_name", "Name"), _col("email", "Email"),
                    _col("te_suid", "TE_SUID"), _col("status_code", "Status", "number"),
                    _col("created_date", "Created", "datetime"), _col("account_id", "ACCOUNT_ID", "number")],
    },
    "user_account": {
        "select": "a.account_id, a.te_suid, a.type_code, a.status_code, a.user_name, a.email_id, a.mobile_number, "
                  "a.invalid_password_count, a.invalid_otp_count, a.mobile_verified, a.email_verified, "
                  "a.created_by, a.created_date, a.updated_by, a.updated_date, a.status_updated_date, a.spouse_grid",
        "from": "user_account a",
        "order": "a.account_id",
        "search": ["a.te_suid", "a.user_name", "a.email_id", "a.mobile_number"],
        "filters": {"te_suid": "a.te_suid", "status_code": "a.status_code"},
        "columns": [_col("account_id", "ACCOUNT_ID", "number"), _col("te_suid", "TE_SUID"),
                    _col("type_code", "TYPE_CODE", "number"), _col("status_code", "STATUS_CODE", "number"),
                    _col("user_name", "USER_NAME"), _col("email_id", "EMAIL_ID"),
                    _col("mobile_number", "MOBILE_NUMBER"),
                    _col("invalid_password_count", "INVALID_PASSWORD_COUNT", "number"),
                    _col("invalid_otp_count", "INVALID_OTP_COUNT", "number"),
                    _col("mobile_verified", "MOBILE_VERIFIED"), _col("email_verified", "EMAIL_VERIFIED"),
                    _col("created_by", "CREATED_BY"), _col("created_date", "CREATED_DATE", "datetime"),
                    _col("updated_by", "UPDATED_BY"), _col("updated_date", "UPDATED_DATE", "datetime"),
                    _col("status_updated_date", "STATUS_UPDATED_DATE", "datetime"),
                    _col("spouse_grid", "SPOUSE_GRID")],
    },
    "epayment_method": {
        "select": "m.epay_method_id, m.user_account_id, a.te_suid, m.nickname, m.epay_mode, m.status, m.card_type, "
                  "m.cardholder_name, m.created_date, m.updated_date, m.last_used_date, m.tc_accepted_date",
        "from": "epayment_method m JOIN user_account a ON a.account_id = m.user_account_id",
        "order": "m.epay_method_id",
        "search": ["a.te_suid", "m.nickname", "m.cardholder_name"],
        "filters": {"te_suid": "a.te_suid", "epay_mode": "m.epay_mode", "status": "m.status"},
        "columns": [_col("epay_method_id", "EPAY_METHOD_ID", "number"),
                    _col("user_account_id", "USER_ACCOUNT_ID", "number"), _col("te_suid", "TE_SUID"),
                    _col("nickname", "NICKNAME"), _col("epay_mode", "EPAY_MODE", "badge"),
                    _col("status", "STATUS", "badge"), _col("card_type", "CARD_TYPE"),
                    _col("cardholder_name", "CARDHOLDER_NAME"), _col("created_date", "CREATED_DATE", "datetime"),
                    _col("updated_date", "UPDATED_DATE", "datetime"),
                    _col("last_used_date", "LAST_USED_DATE", "datetime"),
                    _col("tc_accepted_date", "TC_ACCEPTED_DATE", "datetime")],
    },
    "epayment": {
        "select": "p.epmt_id, p.te_suid, p.src, p.amt, p.crcrd_rsp_suid, p.crcrd_cnf_nbr, p.crcrd_txn_id, "
                  "p.epmt_timstmp, p.epmt_status, p.bat_nbr, p.gateway_suid, p.gateway_rank, p.result, "
                  "p.bank_account_id, p.echeck_cnf_nbr, p.internal_user_id, p.extract_timstmp, p.crcrd_type, "
                  "p.ems_payment_id, p.response_message, p.epay_method_id",
        "from": "epayment p",
        "order": "p.epmt_timstmp DESC, p.epmt_id DESC",
        "search": ["p.te_suid", "p.crcrd_cnf_nbr", "p.crcrd_txn_id", "p.echeck_cnf_nbr", "p.response_message"],
        "filters": {"te_suid": "p.te_suid", "epmt_status": "p.epmt_status", "result": "p.result"},
        "columns": [_col("epmt_id", "EPMT_ID", "number"), _col("te_suid", "TE_SUID"), _col("src", "SRC"),
                    _col("amt", "AMT", "money"), _col("crcrd_rsp_suid", "CRCRD_RSP_SUID"),
                    _col("crcrd_cnf_nbr", "CRCRD_CNF_NBR"), _col("crcrd_txn_id", "CRCRD_TXN_ID"),
                    _col("epmt_timstmp", "EPMT_TIMSTMP", "datetime"), _col("epmt_status", "EPMT_STATUS", "badge"),
                    _col("bat_nbr", "BAT_NBR"), _col("gateway_suid", "GATEWAY_SUID"),
                    _col("gateway_rank", "GATEWAY_RANK"), _col("result", "RESULT", "badge"),
                    _col("bank_account_id", "BANK_ACCOUNT_ID"), _col("echeck_cnf_nbr", "ECHECK_CNF_NBR"),
                    _col("internal_user_id", "INTERNAL_USER_ID"),
                    _col("extract_timstmp", "EXTRACT_TIMSTMP", "datetime"), _col("crcrd_type", "CRCRD_TYPE"),
                    _col("ems_payment_id", "EMS_PAYMENT_ID"),
                    _col("response_message", "RESPONSE_MESSAGE", "long"),
                    _col("epay_method_id", "EPAY_METHOD_ID", "number")],
    },
    "conversations": {
        "select": "c.id, ua.user_name AS username, c.session_id, c.title, c.status, "
                  "(SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count, "
                  "c.created_at, c.updated_at",
        "from": "conversations c JOIN user_account ua ON ua.account_id = c.account_id",
        "order": "c.updated_at DESC, c.id",
        "search": ["ua.user_name", "c.title", "c.session_id"],
        "filters": {"username": "ua.user_name"},
        "columns": [_col("username", "User"), _col("title", "Title"), _col("status", "Status", "badge"),
                    _col("message_count", "Messages", "number"), _col("updated_at", "Updated", "datetime"),
                    _col("session_id", "Session", "id"), _col("id", "ID", "id")],
    },
    "messages": {  # the conversation_message table
        "select": "m.id, c.id AS conversation_id, ua.user_name AS username, c.session_id, m.sequence_number, m.role, "
                  f"LEFT(m.content, {PREVIEW_CHARS}) AS content, m.sources, m.created_at",
        "from": "messages m JOIN conversations c ON c.id = m.conversation_id "
                "JOIN user_account ua ON ua.account_id = c.account_id",
        "order": "m.id DESC",
        "search": ["ua.user_name", "m.content"],
        "filters": {"username": "ua.user_name", "conversation_id": "m.conversation_id", "role": "m.role"},
        "columns": [_col("created_at", "Time", "datetime"), _col("username", "User"), _col("role", "Role", "badge"),
                    _col("content", "Message", "long"), _col("sources", "Sources"),
                    _col("sequence_number", "#", "number"), _col("conversation_id", "Conversation", "id")],
    },
    "agent_flows": {
        "select": "t.id, t.username, t.session_id, t.question, t.final_agent, t.sources, t.duration_ms, "
                  "(SELECT COUNT(*) FROM agent_events e WHERE e.turn_id = t.id) AS steps, "
                  "(SELECT COUNT(*) FROM agent_events e WHERE e.turn_id = t.id AND e.event_type = 'tool_call') "
                  "AS tool_calls, t.created_at",
        "from": "agent_turns t",
        "order": "t.created_at DESC, t.id",
        "search": ["t.username", "t.question", "t.final_agent"],
        "filters": {"username": "t.username", "final_agent": "t.final_agent"},
        "columns": [_col("created_at", "Time", "datetime"), _col("username", "User"),
                    _col("question", "Question", "long"), _col("final_agent", "Assigned to", "badge"),
                    _col("steps", "Steps", "number"), _col("tool_calls", "Tool calls", "number"),
                    _col("duration_ms", "ms", "number"), _col("sources", "Sources")],
    },
    "agent_activity": {
        "select": "e.id, e.created_at, t.username, e.turn_id, e.step, e.event_type, e.from_actor, e.to_actor, "
                  "e.tool_name, e.storage_backend, e.status, e.duration_ms, e.detail",
        "from": "agent_events e JOIN agent_turns t ON t.id = e.turn_id",
        "order": "e.created_at DESC, e.id DESC",
        "search": ["t.username", "t.question", "e.event_type", "e.from_actor", "e.to_actor", "e.tool_name", "e.storage_backend", "e.detail"],
        "filters": {"username": "t.username", "event_type": "e.event_type", "backend": "e.storage_backend", "turn_id": "e.turn_id"},
        "columns": [_col("created_at", "Time", "datetime"), _col("username", "User"), _col("event_type", "Event", "badge"),
                    _col("from_actor", "From", "badge"), _col("to_actor", "To", "badge"),
                    _col("tool_name", "Tool function"), _col("storage_backend", "Storage", "badge"),
                    _col("status", "Status", "badge"), _col("duration_ms", "ms", "number"), _col("detail", "Detail", "long")],
    },
    "tool_calls": {
        "select": "e.id, e.created_at, t.username, e.turn_id, "
                  "CASE WHEN e.event_type = 'tool_result' THEN e.to_actor ELSE e.from_actor END AS caller, "
                  "e.tool_name AS tool, "
                  "CASE WHEN e.event_type = 'tool_result' THEN e.from_actor ELSE e.to_actor END AS receiver, "
                  "e.status, e.duration_ms, e.detail",
        "from": "agent_events e JOIN agent_turns t ON t.id = e.turn_id",
        "where": ["e.tool_name IS NOT NULL", "e.event_type IN ('route', 'tool_result')"],
        "order": "e.id DESC",
        "search": ["t.username", "e.tool_name", "e.detail"],
        "filters": {"username": "t.username", "tool": "e.tool_name", "status": "e.status", "turn_id": "e.turn_id"},
        "columns": [_col("created_at", "Time", "datetime"), _col("username", "User"),
                    _col("caller", "Called by", "badge"), _col("tool", "Tool function"),
                    _col("receiver", "Received by", "badge"), _col("status", "Status", "badge"),
                    _col("duration_ms", "ms", "number"), _col("detail", "Detail", "long"),
                    _col("turn_id", "Flow", "id")],
    },
}

_EMBEDDING_COLLECTIONS = {
    "uploaded_documents": ("Uploaded documents", "filename", "text"),
}


def _jsonable(value):
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return value


def _rows(rows: list[dict]) -> list[dict]:
    return [{k: _jsonable(v) for k, v in row.items()} for row in rows]


def _require_mysql() -> None:
    if not db.enabled():
        raise HTTPException(status_code=503, detail="MySQL is not configured")


def _like(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


@router.get("/me")
def me(admin: str = Depends(admin_user)):
    return {"username": admin, "is_admin": True}


@router.get("/valkey-cache")
def valkey_cache_entries(
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0, le=1_000_000),
    q: str = Query("", max_length=MAX_FILTER_CHARS),
):
    """Questions cached in Valkey with their embeddings (the FAQ sheet plus questions learned from Qdrant)."""
    from mcp_server import cache

    try:
        rows = cache.inspect_entries()
    except Exception:
        logger.exception("Valkey inspection failed")
        raise HTTPException(status_code=503, detail="Valkey is unavailable")
    needle = q.strip().lower()
    if needle:
        rows = [r for r in rows if needle in r["question"].lower() or needle in r["answer"].lower()]
    return {"total": len(rows), "limit": limit, "offset": offset, "rows": rows[offset:offset + limit]}


@router.get("/valkey-cache/entry")
def valkey_cache_entry(key: str = Query(..., max_length=100)):
    from mcp_server import cache

    try:
        entry = cache.get_entry(key)
    except Exception:
        logger.exception("Valkey read failed")
        raise HTTPException(status_code=503, detail="Valkey is unavailable")
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown cache entry")
    return entry


@router.get("/tables/{name}")
def table(
    name: str,
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0, le=1_000_000),
    q: str = Query("", max_length=MAX_FILTER_CHARS),
):
    spec = _TABLES.get(name)
    if spec is None:
        raise HTTPException(status_code=404, detail="Unknown table")
    _require_mysql()
    where, params = list(spec.get("where", [])), []
    if q.strip():
        where.append("(" + " OR ".join(f"{c} LIKE %s" for c in spec["search"]) + ")")
        params += [_like(q.strip())] * len(spec["search"])
    for key, column in spec["filters"].items():
        value = request.query_params.get(key)
        if value:
            where.append(f"{column} = %s")
            params.append(value[:MAX_FILTER_CHARS])
    clause = f" WHERE {' AND '.join(where)}" if where else ""
    try:
        with db.connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM {spec['from']}{clause}", params)
            total = cur.fetchone()["n"]
            cur.execute(
                f"SELECT {spec['select']} FROM {spec['from']}{clause} ORDER BY {spec['order']} LIMIT %s OFFSET %s",
                [*params, limit, offset],
            )
            rows = _rows(cur.fetchall())
    except Exception:
        logger.exception("Admin table query failed: %s", name)
        raise HTTPException(status_code=503, detail="The database is unavailable")
    return {"name": name, "columns": spec["columns"], "rows": rows, "total": total, "limit": limit, "offset": offset}


@router.get("/conversations/{conversation_id}/history")
def conversation_history(conversation_id: str):
    """The full transcript of one conversation, each question linked to its recorded agent flow."""
    _require_mysql()
    try:
        uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid conversation id")
    try:
        with db.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT c.id, c.session_id, c.title, c.status, c.created_at, c.updated_at, "
                "ua.user_name AS username "
                "FROM conversations c JOIN user_account ua ON ua.account_id = c.account_id WHERE c.id = %s",
                (conversation_id,),
            )
            conversation = cur.fetchone()
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            cur.execute(
                "SELECT id, sequence_number, role, content, sources, created_at FROM messages "
                "WHERE conversation_id = %s ORDER BY sequence_number LIMIT %s",
                (conversation_id, HISTORY_MAX_MESSAGES),
            )
            messages = cur.fetchall()
            cur.execute(
                "SELECT id, question, created_at FROM agent_turns WHERE username = %s AND session_id = %s "
                "ORDER BY created_at LIMIT %s",
                (conversation["username"], conversation["session_id"], HISTORY_MAX_MESSAGES),
            )
            turns = cur.fetchall()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Admin history query failed")
        raise HTTPException(status_code=503, detail="The database is unavailable")
    unused = list(turns)
    out = []
    for m in messages:
        item = {
            "id": m["id"], "sequence_number": m["sequence_number"], "role": m["role"], "content": m["content"],
            "sources": json.loads(m["sources"]) if m["sources"] else [], "created_at": _jsonable(m["created_at"]),
            "flow_id": None,
        }
        if m["role"] == "user":  # the flow recorded for this very question, matched by text and nearest time
            matches = [t for t in unused if t["question"] == m["content"][:2000]]
            if matches:
                best = min(matches, key=lambda t: abs((t["created_at"] - m["created_at"]).total_seconds()))
                item["flow_id"] = best["id"]
                unused.remove(best)
        out.append(item)
    return {"conversation": _rows([conversation])[0], "messages": out, "truncated": len(messages) >= HISTORY_MAX_MESSAGES}


@router.get("/flows/{turn_id}")
def flow(turn_id: str):
    """One question and every step it went through: who handed it to whom and which tool function ran."""
    _require_mysql()
    try:
        uuid.UUID(turn_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid flow id")
    try:
        with db.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, session_id, question, final_agent, sources, duration_ms, created_at "
                "FROM agent_turns WHERE id = %s",
                (turn_id,),
            )
            turn = cur.fetchone()
            if turn is None:
                raise HTTPException(status_code=404, detail="Flow not found")
            cur.execute(
                "SELECT step, event_type, from_actor, to_actor, tool_name, status, storage_backend, detail, elapsed_ms, duration_ms "
                "FROM agent_events WHERE turn_id = %s ORDER BY step",
                (turn_id,),
            )
            events = cur.fetchall()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Admin flow query failed")
        raise HTTPException(status_code=503, detail="The database is unavailable")
    return {"turn": _rows([turn])[0], "events": _rows(events)}


@router.get("/flow-registry")
def flow_registry():
    """Which tool functions each agent can use, and who can assign work to whom."""
    known = set(ALLOWED_TOOLS) | {"route_request"}
    registry = {
        agent: {**info, "tools": [{"name": t, "known": t in known} for t in info["tools"]]}
        for agent, info in flow_store.AGENT_REGISTRY.items()
    }
    return {"agents": registry, "mcp_tools": sorted(ALLOWED_TOOLS)}


# --- embeddings (Qdrant) ----------------------------------------------------------------------------------------

def _qdrant():
    if not vectordb.enabled():
        raise HTTPException(status_code=503, detail="The vector database is not configured")
    return vectordb.client()


def _clip(value, limit: int = PREVIEW_CHARS):
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _embedding_row(collection: str, point) -> dict:
    _, title_key, text_key = _EMBEDDING_COLLECTIONS[collection]
    payload = point.payload or {}
    vector = point.vector or []
    title = payload.get(title_key) or payload.get("question") or ""
    meta = {k: v for k, v in payload.items() if k not in (title_key, text_key, "text", "content_hash") and not
            isinstance(v, (list, dict)) and v is not None}
    return {
        "id": str(point.id),
        "title": _clip(title, 120),
        "text": _clip(payload.get(text_key) or "(no answer, a member-specific question)", PREVIEW_CHARS),
        "meta": _clip("; ".join(f"{k}={v}" for k, v in meta.items()), 160),
        "dimensions": len(vector),
        "vector_preview": [round(float(x), 4) for x in vector[:6]],
    }


@router.get("/embeddings")
def embedding_collections():
    client = _qdrant()
    out = []
    try:
        for name, (label, _, _) in _EMBEDDING_COLLECTIONS.items():
            if not client.collection_exists(name):
                out.append({"name": name, "label": label, "points": 0, "dimensions": None, "distance": None})
                continue
            params = client.get_collection(name).config.params.vectors
            out.append({
                "name": name, "label": label, "points": client.count(name, exact=True).count,
                "dimensions": params.size, "distance": str(params.distance.value if hasattr(params.distance, "value") else params.distance),
            })
    except Exception:
        logger.exception("Admin embeddings query failed")
        raise HTTPException(status_code=503, detail="The vector database is unavailable")
    return {"collections": out}


@router.get("/embeddings/{collection}")
def embedding_points(collection: str, limit: int = Query(25, ge=1, le=100), cursor: str | None = Query(None, max_length=64)):
    if collection not in _EMBEDDING_COLLECTIONS:
        raise HTTPException(status_code=404, detail="Unknown collection")
    if cursor is not None:
        try:
            uuid.UUID(cursor)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid cursor")
    client = _qdrant()
    try:
        if not client.collection_exists(collection):
            return {"collection": collection, "rows": [], "total": 0, "next_cursor": None}
        points, next_cursor = client.scroll(collection, limit=limit, offset=cursor, with_payload=True, with_vectors=True)
        total = client.count(collection, exact=True).count
    except Exception:
        logger.exception("Admin embeddings scroll failed")
        raise HTTPException(status_code=503, detail="The vector database is unavailable")
    return {
        "collection": collection, "total": total, "next_cursor": str(next_cursor) if next_cursor else None,
        "rows": [_embedding_row(collection, p) for p in points],
    }


@router.get("/embeddings/{collection}/{point_id}")
def embedding_point(collection: str, point_id: str):
    """The full stored text, its metadata and the whole embedding vector."""
    if collection not in _EMBEDDING_COLLECTIONS:
        raise HTTPException(status_code=404, detail="Unknown collection")
    try:
        uuid.UUID(point_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid point id")
    client = _qdrant()
    try:
        found = client.retrieve(collection, [point_id], with_payload=True, with_vectors=True)
    except Exception:
        logger.exception("Admin embedding lookup failed")
        raise HTTPException(status_code=503, detail="The vector database is unavailable")
    if not found:
        raise HTTPException(status_code=404, detail="Point not found")
    point = found[0]
    vector = [float(x) for x in (point.vector or [])]
    return {"id": str(point.id), "collection": collection, "payload": point.payload, "dimensions": len(vector), "vector": vector}


# --- create endpoints -------------------------------------------------------------------------------------------

class CreateUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    email: str = Field(min_length=1, max_length=254)
    display_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=8, max_length=72)


@router.post("/tables/user_account")
def create_user_account(body: CreateUserRequest, admin: str = Depends(admin_user)):
    """Create a new participant: USER_ACCOUNT + users_profile. Returns the new account_id."""
    _require_mysql()
    from mcp_server import db
    ok = db.create_user(body.username, body.email, body.display_name, hash_password(body.password))
    if not ok:
        raise HTTPException(status_code=409, detail="Username or email already exists")
    # Return the created user's account_id
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT account_id FROM user_account WHERE user_name = %s LIMIT 1", (body.username,))
        row = cur.fetchone()
    return {"account_id": row["account_id"], "username": body.username}


class StagePaymentRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    amount: float = Field(gt=0)
    epay_method_id: int | None = None
    due_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


@router.post("/tables/epayment")
def stage_payment(body: StagePaymentRequest, admin: str = Depends(admin_user)):
    """Stage a new premium payment (EPMT_STATUS 'P') for a member. Returns the new payment."""
    _require_mysql()
    from mcp_server import db
    payment = db.stage_payment(body.username, body.amount, body.epay_method_id, body.due_date)
    if payment is None:
        raise HTTPException(status_code=404, detail="Member not found")
    return payment


# --- policy document upload -----------------------------------------------------------

import hashlib
import os
import re
from fastapi import File, UploadFile

MAX_POLICY_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB
ALLOWED_POLICY_EXTENSIONS = (".pdf",)


class PolicyUploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    characters: int
    status: str
    message: str | None = None


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    data = bytearray()
    while chunk := await file.read(64 * 1024):
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(status_code=413, detail=f"File is larger than {limit} bytes")
    return bytes(data)


def _sanitize_filename(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]", "_", os.path.basename(filename.replace("\\", "/")))[:200]


@router.post("/documents/upload-policy", response_model=PolicyUploadResponse)
async def upload_policy_document(file: UploadFile = File(...), admin: str = Depends(admin_user)):
    """
    Upload a policy PDF document, parse it with agentic chunking (Gemini),
    embed chunks and store in Qdrant 'uploaded_documents' collection.
    """
    if not vectordb.enabled():
        raise HTTPException(status_code=503, detail="The vector database is not configured")

    filename = _sanitize_filename(file.filename or "")
    if not filename.lower().endswith(ALLOWED_POLICY_EXTENSIONS):
        raise HTTPException(status_code=415, detail="Only .pdf files are accepted for policy documents")

    data = await _read_limited(file, MAX_POLICY_UPLOAD_BYTES)
    if not data:
        raise HTTPException(status_code=422, detail="The file is empty")

    from app import documents as pipeline
    from mcp_server import documents
    try:
        result = await asyncio.to_thread(pipeline.process, filename, data)
    except documents.DocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("Policy document ingestion failed")
        raise HTTPException(status_code=503, detail="The vector database is unavailable")

    return PolicyUploadResponse(
        doc_id=result["doc_id"],
        filename=filename,
        chunks=result["chunks"],
        characters=result["characters"],
        status=result["status"],
        message=f"Successfully uploaded and indexed {result['chunks']} chunks"
    )


@router.delete("/documents/{collection}/{point_id}")
def delete_document(collection: str, point_id: str, admin: str = Depends(admin_user)):
    """Delete a document chunk from an embedding collection."""
    if collection not in _EMBEDDING_COLLECTIONS:
        raise HTTPException(status_code=404, detail="Unknown collection")
    if not vectordb.enabled():
        raise HTTPException(status_code=503, detail="The vector database is not configured")
    try:
        uuid.UUID(point_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid point id")
    client = vectordb.client()
    try:
        client.delete(collection, [point_id])
    except Exception:
        logger.exception("Qdrant delete failed")
        raise HTTPException(status_code=503, detail="The vector database is unavailable")
    return {"status": "deleted", "point_id": point_id, "collection": collection}


@router.delete("/documents/{collection}")
def delete_all_documents(collection: str, admin: str = Depends(admin_user)):
    """Delete all vectors from an embedding collection."""
    if collection not in _EMBEDDING_COLLECTIONS:
        raise HTTPException(status_code=404, detail="Unknown collection")
    if not vectordb.enabled():
        raise HTTPException(status_code=503, detail="The vector database is not configured")
    client = vectordb.client()
    try:
        # Get all point IDs and delete them
        count_result = client.count(collection, exact=True)
        total = count_result.count
        if total == 0:
            return {"status": "empty", "collection": collection, "deleted": 0}

        # Scroll through all points and delete in batches
        deleted = 0
        offset = None
        batch_size = 100
        while True:
            points, next_offset = client.scroll(collection, limit=batch_size, offset=offset, with_payload=False, with_vectors=False)
            if not points:
                break
            ids = [str(p.id) for p in points]
            client.delete(collection, ids)
            deleted += len(ids)
            if next_offset is None:
                break
            offset = next_offset
    except Exception:
        logger.exception("Qdrant bulk delete failed")
        raise HTTPException(status_code=503, detail="The vector database is unavailable")
    return {"status": "deleted", "collection": collection, "deleted": deleted}
