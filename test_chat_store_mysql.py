import pytest

from app.agents import chat_store, sessions
from mcp_server import db

TEST_DB = "plan_assistant_test"


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
    with db.connection() as conn, conn.cursor() as cur:
        for name in ("alice", "bob"):
            cur.execute(
                "INSERT IGNORE INTO user_account (type_code, status_code, user_name, email_id, enc_password, te_suid) "
                "VALUES (1, 1, %s, %s, 'x', %s)",
                (name, f"{name}@example.test", name),
            )
    yield
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS `{TEST_DB}`")
    mp.undo()


@pytest.fixture(autouse=True)
def clean(mysql_test_db):
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM conversations")
    yield


def test_turns_are_stored_in_order_with_title():
    sessions.append_turn("alice", "s1", "What is my balance?", "It is 10", ["accounts"], None)
    sessions.append_turn("alice", "s1", "Thanks", "Welcome", [], None)
    hist = sessions.get_history("alice", "s1")
    assert [m["role"] for m in hist] == ["user", "assistant", "user", "assistant"]
    assert hist[1]["sources"] == ["accounts"] and hist[0]["time"] > 0
    assert sessions.list_sessions("alice")[0]["title"] == "What is my balance?"
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT sequence_number FROM messages ORDER BY sequence_number")
        assert [r["sequence_number"] for r in cur.fetchall()] == [1, 2, 3, 4]


def test_memory_keeps_pending_quote():
    quote = {"quote_id": "q1", "amount": 5.0, "currency": "USD"}
    sessions.append_turn("alice", "s1", "pay", "confirm?", [], quote)
    assert sessions.load("alice", "s1")["pending_quote"] == quote
    sessions.append_turn("alice", "s1", "confirm", "done", [], None)
    assert sessions.load("alice", "s1")["pending_quote"] is None


def test_sessions_are_isolated_between_users():
    sessions.append_turn("alice", "shared", "secret", "reply", [], None)
    assert sessions.get_history("bob", "shared") == []
    assert sessions.load("bob", "shared")["history"] == []
    assert sessions.list_sessions("bob") == []
    assert not sessions.rename("bob", "shared", "hacked")
    assert not sessions.delete("bob", "shared")
    sessions.append_turn("bob", "shared", "mine", "ok", [], None)
    assert sessions.get_history("alice", "shared")[0]["content"] == "secret"


def test_load_limits_context_but_history_is_complete():
    for i in range(15):
        sessions.append_turn("alice", "long", f"m{i}", f"r{i}", [], None)
    assert len(sessions.load("alice", "long")["history"]) == chat_store.MAX_CONTEXT
    assert len(sessions.get_history("alice", "long")) == 30
    assert sessions.load("alice", "long")["history"][-1]["content"] == "r14"


def test_rename_and_delete_cascade():
    sessions.append_turn("alice", "s1", "hello", "hi", [], {"quote_id": "x", "amount": 1, "currency": "USD"})
    assert sessions.rename("alice", "s1", "  My   plan  ")
    assert sessions.list_sessions("alice")[0]["title"] == "My plan"
    assert sessions.delete("alice", "s1")
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT (SELECT COUNT(*) FROM messages) AS m, (SELECT COUNT(*) FROM conversation_memory) AS c")
        assert dict(cur.fetchone()) == {"m": 0, "c": 0}


def test_list_orders_most_recent_first():
    sessions.append_turn("alice", "old", "first", "r", [], None)
    sessions.append_turn("alice", "new", "second", "r", [], None)
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE conversations SET updated_at = NOW() - INTERVAL 1 HOUR WHERE session_id = 'old'")
    assert [s["session_id"] for s in sessions.list_sessions("alice")] == ["new", "old"]


def test_unknown_user_is_not_persisted():
    sessions.append_turn("ghost", "s1", "hi", "hello", [], None)
    assert sessions.list_sessions("ghost") == []
