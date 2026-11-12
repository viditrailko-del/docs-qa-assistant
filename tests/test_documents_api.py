from sqlalchemy import func, select

from app.db.models import Chunk

MD = b"""---
title: Expenses
---

## Books

Limit is set to $60/year.

## Internet

Policy is set at $100.
"""


def upload(client, data=MD, name="finance/expenses.md", **form):
    return client.post("/documents", files={"file": (name.split("/")[-1], data, "text/markdown")},
                       data={"path": name, **form})


def test_upload_creates_document_and_chunks(client, db, fake_embedder):
    r = upload(client, source_url="https://handbook.example/finance/expenses")
    assert r.status_code == 201
    body = r.json()
    assert body["title"] == "Expenses"
    assert body["path"] == "finance/expenses.md"
    assert body["source_url"] == "https://handbook.example/finance/expenses"
    assert body["chunk_count"] == 2

    chunks = client.get(f"/documents/{body['id']}/chunks").json()
    assert [c["heading_path"] for c in chunks] == ["Expenses > Books", "Expenses > Internet"]
    assert chunks[0]["text"] == "Limit is set to $60/year."
    assert all(c["embed_model"] == "fake-embedder" for c in chunks)
    stored = db.scalars(select(Chunk).order_by(Chunk.seq)).all()
    assert len(stored[0].embedding) == 384


def test_same_file_twice_is_idempotent(client, db, fake_embedder):
    first = upload(client)
    second = upload(client, name="other/name.md")  # same bytes, different name
    assert (first.status_code, second.status_code) == (201, 200)
    assert first.json()["id"] == second.json()["id"]
    assert db.scalar(select(func.count()).select_from(Chunk)) == 2
    assert fake_embedder.passage_calls == 1  # no wasted embedding work on the duplicate


def test_changed_file_is_a_new_document(client):
    a = upload(client).json()
    b = upload(client, data=MD + b"\nOne more line.\n").json()
    assert a["id"] != b["id"]
    assert len(client.get("/documents").json()) == 2


def test_rejects_non_markdown(client):
    assert upload(client, name="notes.pdf").status_code == 415


def test_rejects_non_utf8(client):
    r = upload(client, data="café".encode("latin-1"))
    assert r.status_code == 422 and "UTF-8" in r.json()["detail"]


def test_rejects_empty_and_noise_only(client):
    assert upload(client, data=b"   \n").status_code == 422
    assert upload(client, data=b"<!-- nothing -->\n{{< youtube \"x\" >}}\n").status_code == 422


def test_rejects_too_large(client):
    assert upload(client, data=b"a" * (1024 * 1024 + 1)).status_code == 413


def test_get_and_list(client):
    doc = upload(client).json()
    assert client.get(f"/documents/{doc['id']}").json()["chunk_count"] == 2
    listed = client.get("/documents").json()
    assert [d["id"] for d in listed] == [doc["id"]] and listed[0]["chunk_count"] == 2


def test_unknown_document_is_404(client):
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/documents/{missing}").status_code == 404
    assert client.get(f"/documents/{missing}/chunks").status_code == 404


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
