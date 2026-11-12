import io
from pathlib import Path

import pytest
from docx import Document
from fastapi.testclient import TestClient

from app.main import app
from mcp_server import agentic_chunking, vectordb

client = TestClient(app)
PDF = Path(__file__).resolve().parents[1] / "app" / "data" / "SAG_AFTRA_RAG_Policy_Reference.pdf"


def headers(user="randy"):
    token = client.post("/api/v1/auth/login", json={"username": user, "password": "password"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def upload(name, data, user="randy"):
    return client.post("/api/v1/documents", files={"file": (name, data)}, headers=headers(user))


def stored(doc_id):
    flt = vectordb.models.Filter(must=[vectordb._eq("doc_id", doc_id)])
    return sorted(vectordb._scroll(vectordb.DOCUMENTS, flt), key=lambda p: p["chunk_index"])


@pytest.fixture(autouse=True)
def admins(monkeypatch):
    monkeypatch.setenv("ADMIN_USERS", "randy")


def test_requires_authentication_and_admin_role():
    assert client.post("/api/v1/documents", files={"file": ("a.txt", b"hello")}).status_code == 401
    assert upload("a.txt", b"hello", user="john").status_code == 403


def test_no_admins_configured_means_nobody_can_upload(monkeypatch):
    monkeypatch.delenv("ADMIN_USERS")
    assert upload("a.txt", b"hello").status_code == 403


def test_txt_is_parsed_embedded_and_stored():
    text = "The grace period lasts thirty days.\n\nRenewal happens every quarter."
    r = upload("policy notes.txt", text.encode())
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "inserted" and body["chunks"] == 1 and body["filename"] == "policy notes.txt"
    point = stored(body["doc_id"])[0]
    assert "grace period" in point["text"] and point["filename"] == "policy notes.txt"


def test_docx_paragraphs_and_tables_are_extracted():
    doc = Document()
    doc.add_paragraph("Dependents are covered after enrollment.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Plan", "Gold"
    buffer = io.BytesIO()
    doc.save(buffer)
    body = upload("benefits.docx", buffer.getvalue()).json()
    text = " ".join(p["text"] for p in stored(body["doc_id"]))
    assert "Dependents are covered" in text and "Plan | Gold" in text


def test_pdf_is_parsed_into_many_chunks():
    r = upload("reference.pdf", PDF.read_bytes())
    assert r.status_code == 201 and r.json()["chunks"] > 10
    assert len(stored(r.json()["doc_id"])) == r.json()["chunks"]


def test_same_file_is_not_inserted_twice():
    assert upload("a.txt", b"Same content here.").json()["status"] == "inserted"
    again = upload("renamed.txt", b"Same content here.").json()
    assert again["status"] == "unchanged" and len(stored(again["doc_id"])) == 1


def test_rejects_unsupported_types_and_unsafe_names():
    assert upload("malware.exe", b"MZ").status_code == 415
    assert upload("noextension", b"text").status_code == 415
    r = upload("../../etc/passwd.txt", b"harmless text")
    assert r.status_code == 201 and r.json()["filename"] == "passwd.txt"


def test_rejects_empty_oversized_and_unreadable_files(monkeypatch):
    assert upload("empty.txt", b"").status_code == 422
    assert upload("blank.txt", b"   \n\n  ").status_code == 422
    assert upload("broken.pdf", b"not a pdf").status_code == 422
    assert upload("broken.docx", b"not a zip").status_code == 422
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "10")
    assert upload("big.txt", b"x" * 11).status_code == 413


def test_vector_database_outage_is_reported(monkeypatch):
    monkeypatch.setattr(vectordb, "ingest_document", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    assert upload("a.txt", b"outage test").status_code == 503


def test_chunking_overlaps_and_respects_the_size():
    text = "\n\n".join(f"Paragraph {n} " + "word " * 60 for n in range(10))
    chunks = agentic_chunking.simple_chunk(text, size=400, overlap=60)
    assert len(chunks) > 3 and all(len(c) <= 400 + 60 for c in chunks)
    assert chunks[1].split()[0] in chunks[0]  # the next chunk starts with text repeated from the previous one
    assert agentic_chunking.simple_chunk("short") == ["short"]
