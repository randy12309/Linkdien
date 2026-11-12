import asyncio

import pytest

from app.agents.mcp_client import call_tool
from mcp_server import embeddings, server, vectordb


def run(coro):
    return asyncio.run(coro)


def set_setting(key, value):
    vectordb.set_setting(key, value)


@pytest.fixture
def restore_settings():
    original = vectordb.get_settings()
    yield
    for key, value in original.items():
        set_setting(key, value)


def test_policy_lookup_is_top_5_similarity_search(capsys):
    r = run(call_tool("get_policy_info", {"topic": "What does P10 say about late payment?"}))
    assert r["status"] == "ok" and "$375" in r["data"]["text"]
    out = capsys.readouterr().out
    assert "[RAG]" in out and out.count("score=") <= vectordb.POLICY_TOP_K


def test_search_returns_at_most_top_k_chunks():
    vectordb.ingest_document("many", "many.txt", [[f"late payment rule number {n}" for n in range(8)]])
    assert len(vectordb.search_policy("late payment rule")) == vectordb.POLICY_TOP_K


def test_nothing_uploaded_means_not_found(monkeypatch):
    monkeypatch.setattr(vectordb, "DOCUMENTS", "no_such_collection")
    assert run(call_tool("get_policy_info", {"topic": "anything"}))["error_type"] == "not_found"


def test_policy_store_outage_is_reported(monkeypatch):
    monkeypatch.setattr(vectordb, "search_policy", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    assert run(call_tool("get_policy_info", {"topic": "P10"}))["error_type"] == "unavailable"

def test_eligibility_numbers_come_from_plan_settings(monkeypatch, restore_settings):
    monkeypatch.setattr(server.db, "enabled", lambda: True)
    monkeypatch.setattr(server.db, "total_paid", lambda user: 1_000.00)  # EPAYMENT.AMT is in major units
    set_setting("eligibility_threshold", "5000")
    set_setting("quarterly_premium", "500")
    principal = server.mint_principal("john")
    d = run(call_tool("get_eligibility", {"principal": principal}))["data"]
    assert d["threshold"] == 5000 and d["remaining"] == 4000 and d["quarters_remaining"] == 8
    assert d["quarterly_premium"] == 500 and d["eligible"] is False


def test_premium_breakdown_uses_database_ratio(restore_settings):
    set_setting("base_coverage_ratio", "0.5")
    d = run(call_tool("explain_premium", {"principal": server.mint_principal("randy")}))["data"]
    assert d["breakdown"] == {"base_coverage": 156.0, "taxes_and_fees": 156.0}


def test_missing_plan_settings_are_an_error_not_a_default(monkeypatch):
    monkeypatch.setattr(vectordb, "get_settings", lambda: {})
    r = run(call_tool("get_eligibility", {"principal": server.mint_principal("john")}))
    assert r["status"] == "error" and r["error_type"] == "unavailable"


def test_embeddings_have_the_column_dimension():
    assert all(len(v) == embeddings.DIMENSIONS for v in embeddings.embed(["a", "b c"]))
