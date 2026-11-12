import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from app import users
from app.agents import graph as agent
from app.agents import trace
from app.main import app
from mcp_server import db, vectordb

TEST_DB = "plan_assistant_admin_test"
client = TestClient(app)


def headers(user="randy"):
    token = client.post("/api/v1/auth/login", json={"username": user, "password": "password"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def get(path, user="randy", **params):
    return client.get(f"/api/v1/admin{path}", headers=headers(user), params=params)


@pytest.fixture(autouse=True)
def admins(monkeypatch):
    monkeypatch.setenv("ADMIN_USERS", "randy")


@pytest.fixture(scope="module")
def mysql_test_db():
    mp = pytest.MonkeyPatch()
    mp.setenv("MYSQL_DATABASE", TEST_DB)
    mp.setenv("DB_ENABLED", "1")
    try:
        db.create_schema()
    except Exception as exc:  # MySQL not available or no credentials: skip rather than fail offline runs
        mp.undo()
        pytest.skip(f"MySQL unavailable: {exc}")
    ids = {}
    # USER_ACCOUNT.ENC_PASSWORD is authoritative when MySQL is on, so the fixture stores a real bcrypt
    # hash of the demo password; login would fail against a placeholder.
    hashed = users.hash_password("password")
    with db.connection() as conn, conn.cursor() as cur:
        for name in ("randy", "john"):
            cur.execute(
                "INSERT INTO user_account (type_code, status_code, user_name, email_id, enc_password, te_suid) "
                "VALUES (1, 1, %s, %s, %s, %s)",
                (name, f"{name}@example.test", hashed, name),
            )
            ids[name] = cur.lastrowid
            cur.execute(
                "INSERT INTO users_profile (te_suid, display_name, email) VALUES (%s, %s, %s)",
                (name, name.title(), f"{name}@example.test"),
            )
        cur.execute(
            "INSERT INTO epayment (te_suid, src, amt, epmt_timstmp, epmt_status, response_message) "
            "VALUES ('john', 'W', 700.00, NOW(), 'P', 'Staged, awaiting backend sync')",
        )
        conv = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO conversations (id, account_id, session_id, title) VALUES (%s, %s, 's1', 'Grace period chat')",
            (conv, ids["john"]),
        )
        cur.execute(
            "INSERT INTO messages (conversation_id, sequence_number, role, content, sources) VALUES "
            "(%s, 1, 'user', 'What is the grace period?', NULL), (%s, 2, 'assistant', '30 days.', '[\"policy:grace period\"]')",
            (conv, conv),
        )
    yield {"conversation": conv, "hash": hashed}
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS `{TEST_DB}`")
    mp.undo()


def test_admin_endpoints_require_an_administrator():
    assert client.get("/api/v1/admin/tables/user_account").status_code == 401
    assert get("/tables/user_account", user="john").status_code == 403
    assert get("/me").json() == {"username": "randy", "is_admin": True}


def test_admins_table_grants_access_and_is_synced_from_config(mysql_test_db, monkeypatch):
    monkeypatch.setenv("ADMIN_USERS", "")
    assert get("/me", user="john").status_code == 403
    monkeypatch.setenv("ADMIN_USERS", "randy,ghost")
    assert db.sync_admins(["randy", "ghost"]) == 1  # ghost is not a user, so it is skipped
    monkeypatch.setenv("ADMIN_USERS", "")
    assert get("/me", user="randy").status_code == 200  # still an admin through the table
    rows = get("/tables/admins", user="randy").json()["rows"]
    assert [r["username"] for r in rows] == ["randy"] and rows[0]["granted_by"] == "ADMIN_USERS"


def test_users_table_never_exposes_password_hashes(mysql_test_db):
    body = get("/tables/user_account").json()
    assert body["total"] == 2 and {r["user_name"] for r in body["rows"]} == {"randy", "john"}
    assert mysql_test_db["hash"] not in str(body) and "enc_password" not in str(body)


def test_profile_payments_conversations_and_messages_tables(mysql_test_db):
    profile = get("/tables/users_profile").json()["rows"][0]
    assert profile["display_name"] in {"Randy", "John"} and profile["te_suid"] in {"randy", "john"}
    payment = get("/tables/epayment").json()["rows"][0]
    assert payment["te_suid"] == "john" and payment["amt"] == 700.0 and payment["epmt_status"] == "P"
    conv = get("/tables/conversations").json()["rows"][0]
    assert conv["title"] == "Grace period chat" and conv["message_count"] == 2
    messages = get("/tables/messages", conversation_id=mysql_test_db["conversation"]).json()
    assert messages["total"] == 2 and {m["role"] for m in messages["rows"]} == {"user", "assistant"}


def test_tables_page_search_and_reject_unknown_names(mysql_test_db):
    assert len(get("/tables/messages", limit=1).json()["rows"]) == 1
    assert get("/tables/user_account", q="joh").json()["total"] == 1
    assert get("/tables/user_account", q="%").json()["total"] == 0  # a LIKE wildcard is matched literally
    assert get("/tables/user_account; DROP TABLE user_account").status_code == 404
    assert get("/tables/secrets").status_code == 404
    assert get("/tables/user_account", limit=1000).status_code == 422


def test_history_returns_the_transcript(mysql_test_db):
    body = get(f"/conversations/{mysql_test_db['conversation']}/history").json()
    assert body["conversation"]["username"] == "john"
    assert [(m["role"], m["content"]) for m in body["messages"]] == [("user", "What is the grace period?"), ("assistant", "30 days.")]
    assert body["messages"][1]["sources"] == ["policy:grace period"]
    assert get("/conversations/not-a-uuid/history").status_code == 422
    assert get(f"/conversations/{uuid.uuid4()}/history").status_code == 404


def test_agent_flow_records_the_assignment_and_tool_calls(mysql_test_db, monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", "1")
    monkeypatch.setenv("GROQ_API_KEY", "x")

    async def to_payment(state):
        return "payment", ""

    monkeypatch.setattr(agent, "_llm_route", to_payment)
    monkeypatch.setattr(agent.llm, "enabled", lambda: True)
    asyncio.run(agent.run_agent("john", "I want to make a payment", "s1"))
    trace.wait_pending()

    turn = get("/tables/agent_flows", username="john").json()["rows"][0]
    assert turn["question"] == "I want to make a payment" and turn["final_agent"] == "payment" and turn["tool_calls"] == 1
    events = get(f"/flows/{turn['id']}").json()["events"]
    path = [(e["event_type"], e["from_actor"], e["to_actor"], e["tool_name"]) for e in events]
    assert path[0] == ("question", "user", "supervisor", None)
    assert ("route", "supervisor", "payment_agent", "route_request") in path
    assert ("tool_call", "payment_agent", "mcp_server", "get_due_payments") in path
    assert ("tool_result", "mcp_server", "payment_agent", "get_due_payments") in path
    assert path[-1][0] == "answer" and [e["step"] for e in events] == list(range(1, len(events) + 1))

    calls = get("/tables/tool_calls", username="john").json()["rows"]
    pairs = {(c["caller"], c["tool"], c["receiver"]) for c in calls}
    assert ("supervisor", "route_request", "payment_agent") in pairs
    assert ("payment_agent", "get_due_payments", "mcp_server") in pairs
    assert all("principal" not in c["detail"] for c in calls)


def test_flow_ids_are_validated(mysql_test_db):
    assert get("/flows/nope").status_code == 422
    assert get(f"/flows/{uuid.uuid4()}").status_code == 404


def test_registry_lists_every_tool_each_agent_may_call():
    body = get("/flow-registry").json()
    assert body["agents"]["account_agent"]["tools"] == [
        {"name": n, "known": True}
        for n in ("get_balance", "explain_premium", "get_eligibility", "list_payment_methods", "list_epayments")
    ]
    assert body["agents"]["supervisor"]["assigns_to"] == ["policy_agent", "account_agent", "payment_agent", "sign_in"]
    assert all(t["known"] for a in body["agents"].values() for t in a["tools"])


def test_embeddings_lists_collections_and_points():
    vectordb.ingest_document("admin-view", "view.txt", [[f"chunk {n} about premiums" for n in range(7)]])
    collections = {c["name"]: c for c in get("/embeddings").json()["collections"]}
    assert collections["uploaded_documents"]["dimensions"] == 384 and collections["uploaded_documents"]["points"] >= 7
    page = get("/embeddings/uploaded_documents", limit=5).json()
    assert len(page["rows"]) == 5 and page["next_cursor"]
    row = page["rows"][0]
    assert row["dimensions"] == 384 and len(row["vector_preview"]) == 6 and row["text"]
    second = get("/embeddings/uploaded_documents", limit=5, cursor=page["next_cursor"]).json()
    assert {r["id"] for r in second["rows"]}.isdisjoint({r["id"] for r in page["rows"]})
    full = get(f"/embeddings/uploaded_documents/{row['id']}").json()
    assert len(full["vector"]) == 384 and full["payload"]["filename"]


def test_embeddings_reject_bad_input():
    assert get("/embeddings/plan_settings").status_code == 404
    assert get("/embeddings/uploaded_documents", cursor="x").status_code == 422
    assert get("/embeddings/uploaded_documents/not-a-uuid").status_code == 422
    assert get(f"/embeddings/uploaded_documents/{uuid.uuid4()}").status_code == 404

def test_embedding_views_are_admin_only():
    assert get("/embeddings", user="john").status_code == 403
