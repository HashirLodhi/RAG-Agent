import app as rag
from fastapi.testclient import TestClient
from langchain_core.embeddings import FakeEmbeddings


def test_health():
    assert TestClient(rag.app).get("/health").json() == {"status": "ok"}


def test_sessions_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "DATA_DIR", tmp_path)
    client = TestClient(rag.app)
    a = client.post("/api/sessions", headers={"X-User-ID": "alice"}, json={}).json()
    b = client.post("/api/sessions", headers={"X-User-ID": "bob"}, json={}).json()
    assert a["id"] != b["id"]
    assert len(client.get("/api/sessions", headers={"X-User-ID": "alice"}).json()) == 1
    assert client.get(f"/api/sessions/{a['id']}/history", headers={"X-User-ID": "bob"}).status_code == 404


def test_rejects_unsafe_user_id():
    response = TestClient(rag.app).get("/api/sessions", headers={"X-User-ID": "../other"})
    assert response.status_code == 400


def test_token_f1():
    assert rag.token_f1("alpha beta", "alpha beta") == 1.0
    assert rag.token_f1("alpha", "beta") == 0.0


def test_pdf_upload_and_page_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "DATA_DIR", tmp_path)
    monkeypatch.setattr(rag, "_embeddings", FakeEmbeddings(size=16))
    client = TestClient(rag.app)
    headers = {"X-User-ID": "upload_test"}
    session = client.post("/api/sessions", headers=headers, json={}).json()
    pdf_path = rag.BASE_DIR / "Terms & Conditions.pdf"
    with pdf_path.open("rb") as pdf:
        response = client.post(
            f"/api/sessions/{session['id']}/documents",
            headers=headers,
            files=[("files", (pdf_path.name, pdf, "application/pdf"))],
        )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["documents"][0]["pages"] > 0
    assert result["total_chunks"] > 0
